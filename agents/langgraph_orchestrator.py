import os
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from agents.patient_agent import run_patient_agent
from agents.doctor_agent import run_doctor_agent
from agents.insurance_agent import run_insurance_agent

load_dotenv()

# General fallback OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AccessDenied(Exception):
    pass


# Rbac enforcement
def enforce_access(state, tool_name):
    """Enforce RBAC, patient identity binding, and consent checks."""
    role = state.get("role", "patient")
    patient_id = state.get("patient_id")
    consent = bool(state.get("consent", True))

    if not patient_id:
        raise AccessDenied("Access denied: missing patient identity binding (patient_id).")

    if tool_name in ["patient_records", "doctor_notes"] and not consent:
        raise AccessDenied("Access denied: patient consent is required.")

    if tool_name == "doctor_notes" and role not in ["doctor", "admin"]:
        raise AccessDenied("Access denied: only doctors can access doctor notes.")

    if tool_name == "insurance" and role not in ["patient", "insurance", "admin"]:
        raise AccessDenied("Access denied: role not permitted for insurance lookup.")


def classify_intent(state):
    msg = (state.get("user_message") or "").lower()

    insurance_kws = ["insurance", "copay", "deductible", "coverage", "claim", "premium"]
    doctor_kws = ["doctor note", "visit note", "clinical note", "physician note"]
    record_kws = [
        "my record", "my chart", "my file",
        "my allergies", "my diagnosis", "my conditions",
        "my labs", "my lab results",
        "my medications", "my medical history"
    ]

    if any(kw in msg for kw in insurance_kws):
        state["intent"] = "insurance"
    elif any(kw in msg for kw in doctor_kws):
        state["intent"] = "doctor_notes"
    elif any(kw in msg for kw in record_kws):
        state["intent"] = "patient_records"
    else:
        state["intent"] = "general"

    return state


def patient_records_node(state):
    enforce_access(state, "patient_records")

    state["reply"] = run_patient_agent(
        user_message=state.get("user_message", ""),
        role=state.get("role", "patient"),
        patient_id=state.get("patient_id", "patient_001"),
        consent=bool(state.get("consent", True)),
    )

    return state


def doctor_notes_node(state):
    enforce_access(state, "doctor_notes")
    pid = state.get("patient_id", "patient_001")
    state["reply"] = run_doctor_agent(pid)
    return state


def insurance_node(state):
    enforce_access(state, "insurance")
    pid = state.get("patient_id", "patient_001")
    state["reply"] = run_insurance_agent(pid)
    return state


def general_node(state):
    user_message = state.get("user_message", "")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful healthcare chatbot. "
                        "Answer the user's question using general medical knowledge. "
                        "If unsure, clearly state uncertainty. "
                        "Do NOT claim access to patient records."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )
        state["reply"] = response.output_text or "I'm sorry, I couldn't generate a response."
    except Exception as e:
        state["reply"] = f"[general] Error contacting OpenAI: {str(e)}"

    return state


def route_from_intent(state):
    return state.get("intent", "general")


graph = StateGraph(dict)

graph.add_node("classify", classify_intent)
graph.add_node("patient_records", patient_records_node)
graph.add_node("doctor_notes", doctor_notes_node)
graph.add_node("insurance", insurance_node)
graph.add_node("general", general_node)

graph.add_edge(START, "classify")

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

graph.add_edge("patient_records", END)
graph.add_edge("doctor_notes", END)
graph.add_edge("insurance", END)
graph.add_edge("general", END)

compiled = graph.compile()


def run_orchestrator(user_message, role="patient", patient_id="patient_001", consent=True):
    state = {
        "user_message": user_message,
        "role": role,
        "patient_id": patient_id,
        "consent": consent,
    }
    result = compiled.invoke(state)
    return result.get("reply", "")
