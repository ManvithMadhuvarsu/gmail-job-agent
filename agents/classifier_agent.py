"""
agents/classifier_agent.py
LangGraph-based agent that:
1. Classifies each email into a category
2. Decides what action to take
3. Generates reply drafts where needed
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

# ── LLM Resilient Setup ───────────────────────────────────────────────────────
def get_resilient_llm():
    """Returns a model with fallback logic: Ollama -> Groq."""
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "claude-opus-4.6")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 1. Try to prepare Ollama if requested
    if use_ollama:
        print(f"  Attempting to use Ollama: {ollama_model} at {ollama_base_url}...")
        try:
            import urllib.request
            req = urllib.request.Request(f"{ollama_base_url.rstrip('/')}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return ChatOllama(model=ollama_model, base_url=ollama_base_url, temperature=0.1)
        except Exception as e:
            print("  ⚠️  Ollama connection failed. Falling back to Groq Cloud...")

    # 2. Fallback to Groq (Cloud)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

llm = get_resilient_llm()

# ── Agent State ───────────────────────────────────────────────────────────────
class EmailState(TypedDict):
    email: dict
    category: str
    action: str
    draft_subject: str
    draft_body: str
    reasoning: str


# ── Prompts ───────────────────────────────────────────────────────────────────
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert email classifier for a job applicant.
Classify the email into EXACTLY one of these categories:

- REJECTION     : Company says no, not moving forward, application declined
- INTERVIEW     : Invitation for interview, screening call, assessment, or next steps
- HOLD          : Application on hold, will be revisited, waitlisted
- FOLLOW_UP     : Recruiter asking for documents, more info, or checking availability  
- APPLIED       : Auto-confirmation that an application was received
- IRRELEVANT    : Spam, promotions, non-job-related emails

Respond with ONLY the category label. Nothing else."""),
    ("human", "Subject: {subject}\nFrom: {sender}\nBody:\n{body}"),
])

ACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are deciding what action to take for a job application email.

Given the category, decide the action:
- REJECTION  → DRAFT_FEEDBACK  (draft a polite feedback request reply)
- INTERVIEW  → DRAFT_CONFIRM   (draft a confirmation/availability reply)  
- HOLD       → LABEL_ONLY      (just label it, no reply needed)
- FOLLOW_UP  → DRAFT_RESPONSE  (draft a helpful response)
- APPLIED    → LABEL_ONLY      (just label it)
- IRRELEVANT → SKIP            (do nothing)

Respond with ONLY the action label. Nothing else."""),
    ("human", "Category: {category}\nSubject: {subject}"),
])

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a seasoned HR Manager and Career Strategist drafting email replies on behalf of a job candidate. 

CRITICAL INSTRUCTION: You MUST carefully read the original email and extract specific details — the company name, the role title, the recruiter's name (if mentioned), and any specific requests or information. Your reply MUST directly reference these details. Generic responses are UNACCEPTABLE.

CANDIDATE DETAILS:
- Name: {name}
- Phone: {phone}
- Email: {email}
- LinkedIn: {linkedin}

═══════════════════════════════════════════════════
DRAFTING TEMPLATES BY ACTION TYPE:
═══════════════════════════════════════════════════

FOR REJECTIONS (DRAFT_FEEDBACK):
Structure your reply in exactly 3 paragraphs:

Paragraph 1 — Graceful Acknowledgment:
- Thank the recruiter BY NAME if mentioned in the email
- Reference the SPECIFIC role and company name from the email
- Acknowledge the decision with maturity and grace

Paragraph 2 — Strategic Feedback Request:
- Ask for 1-2 specific areas where you could strengthen your candidacy
- Frame this as: "Could you share whether the gap was primarily in [technical depth / domain experience / cultural alignment]?"
- Mention that this feedback would be invaluable for your professional development

Paragraph 3 — Forward-Looking Close:
- Express continued admiration for the company's work
- Request to be considered for future opportunities that align with your profile
- Professional sign-off with full contact details

FOR INTERVIEW INVITES (DRAFT_CONFIRM):
Structure your reply in exactly 3 paragraphs:

Paragraph 1 — Enthusiastic Acknowledgment:
- Express genuine excitement about the opportunity
- Reference the SPECIFIC role title and company from the email
- Thank them for considering your candidacy

Paragraph 2 — Actionable Details:
- Confirm your availability (mention "I am available on [weekdays] between [9 AM - 6 PM IST]" or similar)
- If they mentioned a specific format (phone/video/in-person), confirm you're prepared for that
- Ask if there are specific topics, case studies, or materials you should prepare

Paragraph 3 — Professional Close:
- Reiterate enthusiasm for the conversation
- Provide your direct contact details for scheduling convenience
- Professional sign-off

FOR FOLLOW-UPS (DRAFT_RESPONSE):
Structure your reply in exactly 2-3 paragraphs:

Paragraph 1 — Prompt Acknowledgment:
- Thank them for reaching out and reference what they're asking about
- Show responsiveness and professionalism

Paragraph 2 — Direct Response:
- Address their specific request (documents, information, availability) clearly and completely
- If they asked for documents, confirm you'll attach them
- If they asked about availability, provide clear time slots

Paragraph 3 (if needed) — Close:
- Express continued interest in the opportunity
- Professional sign-off with contact details

═══════════════════════════════════════════════════
ABSOLUTE RULES:
═══════════════════════════════════════════════════
1. LENGTH: 150-250 words. Multi-paragraph ALWAYS. Never a single block.
2. CONTEXT-AWARE: Extract and use the company name, role title, and recruiter name from the original email. NEVER use placeholders like [Company Name].
3. SPECIFIC: Reference actual content from the original email — don't write generic replies.
4. TONE: Professional, warm, articulate. Use vocabulary like "appreciate the transparency", "constructive insights", "strategic alignment", "valuable perspective".
5. SIGNATURE: Always end with:
   
   Best regards,
   [Candidate Name]
   [Phone] | [Email]
   [LinkedIn]
   
6. BANNED PHRASES: "I hope this email finds you well", "To whom it may concern", any placeholder brackets like [X]."""),
    ("human", """Action: {action}
Original Subject: {subject}
From: {sender}
Original Email Body:
{body}

Write the reply email body ONLY. No subject line. Follow the exact paragraph structure for this action type."""),
])



# ── Invocation Wrapper with Fallback ──────────────────────────────────────────
_fallback_llm = None

def get_fallback_model():
    global _fallback_llm
    if _fallback_llm is None:
        _fallback_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    return _fallback_llm

def safe_invoke(chain_or_llm, input_data):
    """Invoke and fallback to Groq if the primary LLM fails."""
    global llm
    try:
        return chain_or_llm.invoke(input_data)
    except Exception as e:
        # If Ollama is failing (missing model, etc.), switch to cloud
        if isinstance(llm, ChatOllama):
            print(f"\n  ⚠️  Primary model (Ollama) failed: {e}")
            print("  🔄  Switching to Groq Cloud fallback for the next retry...")
            llm = get_fallback_model()
            # We don't try to invoke here because input_data might be a dict 
            # while the raw 'llm' expects messages/prompts. 
            # Re-raise so the 'main.py' retry loop can start fresh with the new global 'llm'.
        raise e

# ── Agent Nodes ───────────────────────────────────────────────────────────────
def classify_node(state: EmailState) -> EmailState:
    """Classify the email category."""
    email = state["email"]
    chain = CLASSIFY_PROMPT | llm
    result = safe_invoke(chain, {
        "subject": email["subject"],
        "sender": email["sender"],
        "body": email["body"],
    })
    category = result.content.strip().upper()

    # Validate — default to IRRELEVANT if unrecognised
    valid = {"REJECTION", "INTERVIEW", "HOLD", "FOLLOW_UP", "APPLIED", "IRRELEVANT"}
    if category not in valid:
        category = "IRRELEVANT"

    return {**state, "category": category}


def decide_action_node(state: EmailState) -> EmailState:
    """Decide what action to take."""
    sender = state["email"]["sender"].lower()
    category = state["category"]
    
    # Check for "do-not-reply" patterns
    is_noreply = any(pattern in sender for pattern in ["noreply", "no-reply", "donotreply", "do-not-reply"])

    chain = ACTION_PROMPT | llm
    result = safe_invoke(chain, {
        "category": category,
        "subject": state["email"]["subject"],
    })
    action = result.content.strip().upper()

    valid_actions = {"DRAFT_FEEDBACK", "DRAFT_CONFIRM", "DRAFT_RESPONSE", "LABEL_ONLY", "SKIP"}
    if action not in valid_actions:
        action = "LABEL_ONLY"

    # Override: Do NOT draft if it's a no-reply address
    if is_noreply and action.startswith("DRAFT_"):
        action = "LABEL_ONLY"

    return {**state, "action": action}


def draft_reply_node(state: EmailState) -> EmailState:
    """Generate a reply draft."""
    email = state["email"]
    chain = DRAFT_PROMPT | llm
    result = safe_invoke(chain, {
        "name": os.getenv("YOUR_NAME", "Your Name"),
        "phone": os.getenv("YOUR_PHONE", ""),
        "email": os.getenv("YOUR_EMAIL", ""),
        "linkedin": os.getenv("YOUR_LINKEDIN", ""),
        "action": state["action"],
        "subject": email["subject"],
        "sender": email["sender"],
        "body": email["body"],
    })

    # Build reply subject
    original_subject = email["subject"]
    if not original_subject.lower().startswith("re:"):
        reply_subject = f"Re: {original_subject}"
    else:
        reply_subject = original_subject

    return {
        **state,
        "draft_subject": reply_subject,
        "draft_body": result.content.strip(),
    }


def skip_node(state: EmailState) -> EmailState:
    """No action needed."""
    return {**state, "action": "SKIP", "draft_body": "", "draft_subject": ""}


# ── Routing Logic ─────────────────────────────────────────────────────────────
def route_action(state: EmailState) -> Literal["draft_reply", "skip"]:
    if state["action"] in {"DRAFT_FEEDBACK", "DRAFT_CONFIRM", "DRAFT_RESPONSE"}:
        return "draft_reply"
    return "skip"


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_classifier_graph():
    graph = StateGraph(EmailState)

    graph.add_node("classify", classify_node)
    graph.add_node("decide_action", decide_action_node)
    graph.add_node("draft_reply", draft_reply_node)
    graph.add_node("skip", skip_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "decide_action")
    graph.add_conditional_edges(
        "decide_action",
        route_action,
        {"draft_reply": "draft_reply", "skip": "skip"},
    )
    graph.add_edge("draft_reply", END)
    graph.add_edge("skip", END)

    return graph.compile()


# Singleton — compile once
classifier = build_classifier_graph()


def process_email(email: dict) -> EmailState:
    """Run a single email through the classifier agent."""
    initial_state: EmailState = {
        "email": email,
        "category": "",
        "action": "",
        "draft_subject": "",
        "draft_body": "",
        "reasoning": "",
    }
    return classifier.invoke(initial_state)
