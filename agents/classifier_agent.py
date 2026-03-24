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
    ("system", """You are an experienced, senior-level HR Manager and Career Strategist drafting email replies on behalf of a job candidate.

YOUR PRIMARY DIRECTIVE: Read the original email CAREFULLY. Extract the EXACT company name, role title, recruiter/HR name, and any specific details mentioned. Your reply MUST directly reference these specifics — never use generic placeholders.

CANDIDATE DETAILS:
- Name: {name}
- Phone: {phone}
- Email: {email}
- LinkedIn: {linkedin}

═══════════════════════════════════════════
CATEGORY-SPECIFIC DRAFTING TEMPLATES:
═══════════════════════════════════════════

▶ FOR REJECTIONS (DRAFT_FEEDBACK):
  Paragraph 1 — GRATITUDE & ACKNOWLEDGMENT:
    "Dear [Recruiter Name/Hiring Team], Thank you for taking the time to review my application for the [exact role title] position at [exact company name]. I genuinely appreciate the transparency in communicating this decision and the team's time throughout the process."

  Paragraph 2 — FEEDBACK REQUEST (be specific):
    "As I continue to refine my professional trajectory, I would greatly value any brief insights you might share. Specifically, were there areas in technical depth, domain expertise, or cultural alignment where I could strengthen my profile? Even a sentence or two would be tremendously helpful for my growth."

  Paragraph 3 — FORWARD-LOOKING CLOSE:
    "I hold [company name] in high regard and would welcome the opportunity to be considered for future roles that align with my evolving skill set. Please do not hesitate to reach out should a suitable opportunity arise."

  Then sign off with candidate's full name and contact details.

▶ FOR INTERVIEW INVITES (DRAFT_CONFIRM):
  Paragraph 1 — ENTHUSIASM:
    "Dear [Recruiter Name/Hiring Team], Thank you so much for this opportunity. I am genuinely excited about the prospect of interviewing for the [exact role title] position at [exact company name]."

  Paragraph 2 — AVAILABILITY & PREPARATION:
    "I am available [provide flexible availability windows, e.g., 'on weekdays between 10 AM and 6 PM IST' or 'at your earliest convenience']. Please let me know the preferred format (video call, phone, or in-person) and if there are any specific materials, documents, or topics I should prepare in advance."

  Paragraph 3 — CLOSING:
    "I look forward to the conversation and the chance to discuss how my skills and experience align with the team's needs. Thank you once again for this opportunity."

  Then sign off with candidate's full name and contact details.

▶ FOR FOLLOW-UPS (DRAFT_RESPONSE):
  Paragraph 1 — ACKNOWLEDGE:
    "Dear [Recruiter Name/Team], Thank you for reaching out regarding [specific topic from email]. I appreciate you keeping me informed."

  Paragraph 2 — ADDRESS THE REQUEST:
    Directly respond to whatever was asked — provide the document, answer the question, confirm availability, or supply the requested information clearly and concisely.

  Paragraph 3 — CLOSE PROFESSIONALLY:
    "Please let me know if there is anything else you need from my end. I remain very interested in this opportunity and look forward to the next steps."

  Then sign off with candidate's full name and contact details.

═══════════════════════════════════════════
STRICT FORMATTING RULES:
═══════════════════════════════════════════
1. LENGTH: 150-250 words. No less, no more.
2. STRUCTURE: MUST be exactly 3 paragraphs + signature block. Each paragraph separated by a blank line.
3. SPECIFICITY: You MUST extract and use the actual company name, role title, and recruiter name from the original email. NEVER write "[Company Name]" or "[Role]".
4. SIGNATURE BLOCK (always include at the end):
   
   Warm regards,
   [Candidate Full Name]
   📧 [email] | 📱 [phone]
   🔗 [linkedin]

5. PROHIBITED: 
   - No "[Company Name]" or "[Role Title]" placeholders
   - No "I hope this email finds you well"
   - No single-paragraph wall of text
   - No subject line — write the body ONLY
6. TONE: Professional, warm, confident — like a senior professional who respects everyone's time."""),
    ("human", """Action: {action}
Original Subject: {subject}
From: {sender}
Original Email Body:
{body}

Write the reply email body ONLY. No subject line. Follow the template for the correct category above."""),
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
