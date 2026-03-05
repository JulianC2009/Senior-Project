import os
import glob
from PyPDF2 import PdfReader

KNOWLEDGE_BASE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")

PATIENT_NAME_MAP = {
    "patient_001": "Maria Santos",
    "patient_002": "James OBrien",
    "patient_003": "Priya Kapoor",
    "patient_004": "Robert Chen",
}


def extract_text_from_pdf(filepath: str) -> str:
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


def read_files_from_folder(folder_path: str, patient_name: str | None = None) -> str | None:
    if not os.path.exists(folder_path):
        return None

    results = []
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*"))):
        filename = os.path.basename(filepath)

        if patient_name and patient_name.lower() not in filename.lower():
            continue

        if filepath.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
            if text:
                results.append(f"--- {filename} ---\n{text}")

        elif filepath.lower().endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        results.append(f"--- {filename} ---\n{text}")
            except Exception as e:
                results.append(f"--- {filename} ---\n[Error reading file: {e}]")

    return "\n\n".join(results) if results else None


def run_insurance_agent(patient_id: str) -> str:
    pid = (patient_id or "").lower().strip()
    patient_name = PATIENT_NAME_MAP.get(pid)

    if not patient_name:
        return f"[insurance] No records found for patient ID: {pid}"

    folder_path = os.path.join(KNOWLEDGE_BASE, "Insurance")
    content = read_files_from_folder(folder_path, patient_name=patient_name)

    if content:
        return f"[insurance] Insurance info for {patient_name}:\n\n{content}"

    return f"[insurance] No insurance records found for {patient_name}."