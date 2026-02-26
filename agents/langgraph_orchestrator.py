from langgraph.graph import StateGraph, START, END

# exception for access control violations
class AccessDenied(Exception):
    pass

# Enforcing RBAC, patient identity binding, and consent checks
def enforce_access(state, tool_name):
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


# Classifies the user's intent basing it on keywords
def classify_intent(state):
    msg = (state.get("user_message") or "").lower()

    if "insurance" in msg or "copay" in msg or "deductible" in msg or "coverage" in msg:
        state["intent"] = "insurance"
        return state

    if "doctor note" in msg or "visit note" in msg or "clinical note" in msg:
        state["intent"] = "doctor_notes"
        return state

    if "record" in msg or "allergy" in msg or "condition" in msg or "diagnosis" in msg:
        state["intent"] = "patient_records"
        return state

    state["intent"] = "fallback"
    return state

# Agent nodes for patient, doctor notes, insurance, and fallback tools with access control enforcement
def patient_records_node(state):
    enforce_access(state, "patient_records")
    pid = state.get("patient_id")
    state["reply"] = f"[patient_records] Returned synthetic record for {pid}."
    return state

def doctor_notes_node(state):
    enforce_access(state, "doctor_notes")
    pid = state.get("patient_id")
    state["reply"] = f"[doctor_notes] Returned synthetic doctor notes for {pid}."
    return state

def insurance_node(state):
    enforce_access(state, "insurance")
    pid = state.get("patient_id")
    state["reply"] = f"[insurance] Returned synthetic insurance info for {pid}."
    return state

def fallback_node(state):
    state["reply"] = "I can help with patient records, doctor notes, or insurance. What do you need?"
    return state

# Determines which node to route to after the intent classification
def route_from_intent(state):
    return state.get("intent", "fallback")
graph = StateGraph(dict)

# Registering nodes and an entry point
graph.add_node("classify", classify_intent)
graph.add_node("patient_records", patient_records_node)
graph.add_node("doctor_notes", doctor_notes_node)
graph.add_node("insurance", insurance_node)
graph.add_node("fallback", fallback_node)
graph.add_edge(START, "classify")

# Conditional routing that is based on intent classification results, with a fallback to a default node
graph.add_conditional_edges(
    "classify",
    route_from_intent,
    {
        "patient_records": "patient_records",
        "doctor_notes": "doctor_notes",
        "insurance": "insurance",
        "fallback": "fallback",
    },
)

# All paths lead to END after execution. Compiling the graph for execution
graph.add_edge("patient_records", END)
graph.add_edge("doctor_notes", END)
graph.add_edge("insurance", END)
graph.add_edge("fallback", END)
compiled = graph.compile()

# Entry function that is called by Server.py, which runs the orchestrator graph and returns the final reply
def run_orchestrator(user_message, role="patient", patient_id="patient_001", consent=True):
    state = {
        "user_message": user_message,
        "role": role,
        "patient_id": patient_id,
        "consent": consent,
    }

    result = compiled.invoke(state)
    return result.get("reply", "")
