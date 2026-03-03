import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from langgraph.graph import StateGraph, START, END

load_dotenv()

# OpenAI client for the General Agent fallback
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to the knowledge base folder (relative to project root)
KNOWLEDGE_BASE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")

# Maps patient_id -> (folder name, patient name for file matching)
PATIENT_MAP = {
    "patient_001": ("Patient_001", "Maria Santos"),
    "patient_002": ("Patient_002", "James OBrien"),
    "patient_003": ("Patient_003", "Priya Kapoor"),
    "patient_004": ("Patient_004", "Robert Chen"),
}


# ── PDF Text Extraction

def extract_text_from_pdf(filepath):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        text = f"[Error reading {filepath}: {e}]"
    return text.strip()


def read_files_from_folder(folder_path, patient_name=None):
    """
    Read all PDFs and TXT files from a folder.
    If patient_name is provided, only read files whose name contains that patient name.
    Returns a combined string of all file contents.
    """
    if not os.path.exists(folder_path):
        return None

    results = []

    for filepath in sorted(glob.glob(os.path.join(folder_path, "*"))):
        filename = os.path.basename(filepath)

        # If filtering by patient name, skip files that don't match
        if patient_name and patient_name.lower() not in filename.lower():
            continue

        if filepath.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
            if text:
                results.append(f"--- {filename} ---\n{text}")

        elif filepath.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    results.append(f"--- {filename} ---\n{text}")

    if not results:
        return None

    return "\n\n".join(results)


# ── Access Control

class AccessDenied(Exception):
    pass


def enforce_access(state, tool_name):
    """Enforce RBAC, patient identity binding, and consent checks."""
    role = state.get("role", "patient")
    patient_id = state.get("patient_id")
    consent = bool(state.get("consent", False))

    if not patient_id:
        raise AccessDenied("Access denied: missing patient identity binding (patient_id).")

    if tool_name in ["patient_records", "doctor_notes"] and not consent:
        raise AccessDenied("Access denied: patient consent is required.")

    if tool_name == "doctor_notes" and role not in ["doctor", "admin"]:
        raise AccessDenied("Access denied: only doctors can access doctor notes.")

    if tool_name == "insurance" and role not in ["patient", "insurance", "admin"]:
        raise AccessDenied("Access denied: role not permitted for insurance lookup.")


# ── Intent Classification

def classify_intent(state):
    """Classify user intent based on keywords."""
    msg = (state.get("user_message") or "").lower()

    if any(kw in msg for kw in ["insurance", "copay", "deductible", "coverage", "claim", "premium"]):
        state["intent"] = "insurance"
    elif any(kw in msg for kw in ["doctor note", "visit note", "clinical note", "physician note",
                                    "capacity assessment", "peer review", "mortality", "controlled substance"]):
        state["intent"] = "doctor_notes"
    elif any(kw in msg for kw in ["record", "allergy", "condition", "diagnosis", "lab result",
                                    "medical history", "medication", "patient"]):
        state["intent"] = "patient_records"
    else:
        state["intent"] = "general"

    return state


# ── Agent Nodes

def patient_records_node(state):
    """Patient Agent: reads from knowledge_base/Patient_XXX/ folder."""
    enforce_access(state, "patient_records")

    pid = state.get("patient_id", "").lower()
    patient_info = PATIENT_MAP.get(pid)

    if not patient_info:
        state["reply"] = f"[patient_records] No records found for patient ID: {pid}"
        return state

    folder_name, patient_name = patient_info
    folder_path = os.path.join(KNOWLEDGE_BASE, folder_name)
    content = read_files_from_folder(folder_path)

    if content:
        state["reply"] = f"[patient_records] Records for {patient_name} ({pid}):\n\n{content}"
    else:
        state["reply"] = f"[patient_records] No records found on file for {patient_name}."

    return state


def doctor_notes_node(state):
    """Doctor Agent: reads from knowledge_base/Doctor/ folder, filtered by patient name."""
    enforce_access(state, "doctor_notes")

    pid = state.get("patient_id", "").lower()
    patient_info = PATIENT_MAP.get(pid)

    if not patient_info:
        state["reply"] = f"[doctor_notes] No records found for patient ID: {pid}"
        return state

    _, patient_name = patient_info
    folder_path = os.path.join(KNOWLEDGE_BASE, "Doctor")
    content = read_files_from_folder(folder_path, patient_name=patient_name)

    if content:
        state["reply"] = f"[doctor_notes] Doctor notes for {patient_name}:\n\n{content}"
    else:
        state["reply"] = f"[doctor_notes] No doctor notes found for {patient_name}."

    return state


def insurance_node(state):
    """Insurance Agent: reads from knowledge_base/Insurance/ folder, filtered by patient name."""
    enforce_access(state, "insurance")

    pid = state.get("patient_id", "").lower()
    patient_info = PATIENT_MAP.get(pid)

    if not patient_info:
        state["reply"] = f"[insurance] No records found for patient ID: {pid}"
        return state

    _, patient_name = patient_info
    folder_path = os.path.join(KNOWLEDGE_BASE, "Insurance")
    content = read_files_from_folder(folder_path, patient_name=patient_name)

    if content:
        state["reply"] = f"[insurance] Insurance info for {patient_name}:\n\n{content}"
    else:
        state["reply"] = f"[insurance] No insurance records found for {patient_name}."

    return state


def general_node(state):
    """
    General Agent: handles queries that don't match patient/doctor/insurance.
    Falls back to OpenAI for general medical questions.
    """
    user_message = state.get("user_message", "")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are a helpful healthcare chatbot. Answer the user's question using your general medical knowledge. If unsure, clearly state uncertainty."},
                {"role": "user", "content": user_message},
            ],
        )
        state["reply"] = response.output_text or "I'm sorry, I couldn't generate a response."
    except Exception as e:
        state["reply"] = f"[general] Error contacting OpenAI: {str(e)}"

    return state


# ── Graph Construction

def route_from_intent(state):
    """Determines which node to route to after intent classification."""
    return state.get("intent", "general")


graph = StateGraph(dict)

# Register nodes
graph.add_node("classify", classify_intent)
graph.add_node("patient_records", patient_records_node)
graph.add_node("doctor_notes", doctor_notes_node)
graph.add_node("insurance", insurance_node)
graph.add_node("general", general_node)

# Entry point
graph.add_edge(START, "classify")

# Conditional routing based on intent
graph.add_conditional_edges(
    "classify",
    route_from_intent,
    {
        "patient_records": "patient_records",
        "doctor_notes": "doctor_notes",
        "insurance": "insurance",
        "general": "general",
    },
)

# All paths lead to END
graph.add_edge("patient_records", END)
graph.add_edge("doctor_notes", END)
graph.add_edge("insurance", END)
graph.add_edge("general", END)

compiled = graph.compile()


# ── Entry Point (called by Server.py) ────────────────────────────────

def run_orchestrator(user_message, role="patient", patient_id="patient_001", consent=True):
    """Run the orchestrator graph and return the final reply."""
    state = {
        "user_message": user_message,
        "role": role,
        "patient_id": patient_id,
        "consent": consent,
    }

    result = compiled.invoke(state)
    return result.get("reply", "")