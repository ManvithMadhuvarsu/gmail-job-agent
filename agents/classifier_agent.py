"""
agents/classifier_agent.py
LangGraph-based agent that:
1. Classifies each email into a category
2. Decides what action to take
3. Generates structured, HR-quality reply drafts
"""

import os
import logging
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()
logger = logging.getLogger(__name__)


def _clip_text(text: str, max_chars: int) -> str:
    """Limit prompt size to save tokens; keep start of message (headers + lead)."""
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text or ""
    return text[:max_chars] + "\n\n[...truncated for token efficiency...]"


def _classify_body_limit() -> int:
    return int(os.getenv("LLM_CLASSIFY_BODY_MAX_CHARS", "").strip() or 3200)


def _draft_body_limit() -> int:
    return int(os.getenv("LLM_DRAFT_BODY_MAX_CHARS", "").strip() or 6000)


def _groq_model() -> str:
    return (os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile").strip()


def _action_from_category(category: str) -> str:
    """Fixed mapping — avoids a second LLM call per email (same rules as former ACTION_PROMPT)."""
    return {
        "REJECTION": "DRAFT_FEEDBACK",
        "INTERVIEW": "DRAFT_CONFIRM",
        "HOLD": "LABEL_ONLY",
        "FOLLOW_UP": "DRAFT_RESPONSE",
        "APPLIED": "LABEL_ONLY",
        "IRRELEVANT": "SKIP",
    }.get(category, "LABEL_ONLY")


# ── LLM Resilient Setup ───────────────────────────────────────────────────────
def get_resilient_llm():
    """Return an LLM: try Ollama first, fall back to Groq Cloud."""
    use_ollama    = os.getenv("USE_OLLAMA", "false").lower() == "true"
    require_ollama = os.getenv("REQUIRE_OLLAMA", "false").lower() == "true"
    ollama_model  = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if use_ollama:
        logger.info(f"Attempting Ollama: {ollama_model} @ {ollama_url}")
        print(f"  Attempting Ollama: {ollama_model} @ {ollama_url}...")
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{ollama_url.rstrip('/')}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    logger.info("Ollama reachable. Using Ollama.")
                    return ChatOllama(model=ollama_model, base_url=ollama_url, temperature=0.1)
        except Exception as e:
            if require_ollama:
                raise RuntimeError(f"Ollama unreachable and REQUIRE_OLLAMA=true: {e}") from e
            logger.warning(f"Ollama unreachable ({e}). Falling back to Groq.")
            print("  ⚠️  Ollama unreachable. Falling back to Groq Cloud...")

    logger.info("Using Groq Cloud LLM.")
    return ChatGroq(
        model=_groq_model(),
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


_primary_llm  = None
_fallback_llm = None


def _get_primary() -> ChatGroq | ChatOllama:
    global _primary_llm
    if _primary_llm is None:
        _primary_llm = get_resilient_llm()
    return _primary_llm


def _get_fallback() -> ChatGroq:
    global _fallback_llm
    if _fallback_llm is None:
        _fallback_llm = ChatGroq(
            model=_groq_model(),
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    return _fallback_llm


def safe_invoke(prompt: ChatPromptTemplate, inputs: dict) -> str:
    """
    Invoke the prompt with the primary LLM; fall back to Groq on any failure.
    Returns the response content as a stripped string.
    """
    global _primary_llm
    llm = _get_primary()
    chain = prompt | llm
    try:
        result = chain.invoke(inputs)
        return result.content.strip()
    except Exception as e:
        if os.getenv("REQUIRE_OLLAMA", "false").lower() == "true":
            raise
        err = str(e).lower()
        # Rate limits: fallback uses the same Groq model — avoid duplicate token spend.
        if "429" in err or "rate_limit" in err:
            raise
        logger.warning(f"Primary LLM failed: {e}. Trying Groq fallback...")
        print(f"\n  ⚠️  Primary model failed: {e}")
        print("  🔄  Switching to Groq fallback...")
        # Reset the primary so next call tries fresh
        _primary_llm = None
        fallback_chain = prompt | _get_fallback()
        result = fallback_chain.invoke(inputs)
        return result.content.strip()


# ── Agent State ───────────────────────────────────────────────────────────────
class EmailState(TypedDict):
    email:         dict
    category:      str
    action:        str
    draft_subject: str
    draft_body:    str
    reasoning:     str


# ── Prompts ───────────────────────────────────────────────────────────────────
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Job-email classifier for applicant {candidate_name}. Output ONE token: REJECTION, INTERVIEW, HOLD, FOLLOW_UP, APPLIED, or IRRELEVANT.

REJECTION: declined, not selected, not moving forward, regret.
INTERVIEW: interview, schedule, screening, assessment, test, next round.
HOLD: under review, shortlist pending, will update later.
FOLLOW_UP: asks candidate for documents, info, availability, salary, references.
APPLIED: application received / submission confirmed (often noreply).
IRRELEVANT: spam, promos, OTP, banks, not job-related.

If unsure, IRRELEVANT. Reply with the single label only."""),
    ("human", "Subject: {subject}\nFrom: {sender}\n\nBody:\n{body}"),
])

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Write a professional reply body for a job seeker. Candidate: {name} | {phone} | {email} | {linkedin}

Rules:
- Use recruiter's first name if obvious; else "Dear Hiring Team,".
- Name the real company and role from the email — no placeholders or brackets.
- 2–3 short paragraphs, ~120–180 words total, warm and concise.
- No "I hope this email finds you well", no "To Whom It May Concern".

Action {action}:
- DRAFT_FEEDBACK: thank them; ask for brief feedback on candidacy; close professionally.
- DRAFT_CONFIRM: confirm interest; offer availability window; ask what to prepare.
- DRAFT_RESPONSE: answer what they asked; offer next steps.

End with:
Best regards,
{name}
{phone} | {email}
{linkedin}"""),
    ("human", """Action: {action}
Subject: {subject}
From: {sender}

Their message:
{body}

Reply body only (no subject line)."""),
])


# ── Agent Nodes ───────────────────────────────────────────────────────────────
def classify_node(state: EmailState) -> EmailState:
    """Classify the email into a category."""
    email = state["email"]
    body = _clip_text(email.get("body") or "", _classify_body_limit())
    response = safe_invoke(CLASSIFY_PROMPT, {
        "candidate_name": os.getenv("YOUR_NAME", "the candidate"),
        "subject": email["subject"],
        "sender":  email["sender"],
        "body":    body,
    })

    # Validate — default to IRRELEVANT if unrecognised
    valid = {"REJECTION", "INTERVIEW", "HOLD", "FOLLOW_UP", "APPLIED", "IRRELEVANT"}
    category = response.upper().strip().split()[0] if response else "IRRELEVANT"  # take first word only
    if category not in valid:
        logger.warning(f"Unexpected category '{response}' → defaulting to IRRELEVANT")
        category = "IRRELEVANT"

    logger.info(f"Classified '{email['subject'][:50]}' → {category}")
    return {**state, "category": category}


def decide_action_node(state: EmailState) -> EmailState:
    """Map category → action in code (no extra LLM call — saves tokens)."""
    action = _action_from_category(state["category"])
    logger.info(f"Action decided: {action} (from category)")
    return {**state, "action": action}


def draft_reply_node(state: EmailState) -> EmailState:
    """Generate a professional reply draft."""
    email = state["email"]
    clipped = _clip_text(email.get("body") or "", _draft_body_limit())

    body = safe_invoke(DRAFT_PROMPT, {
        "name":     os.getenv("YOUR_NAME",     "Your Name"),
        "phone":    os.getenv("YOUR_PHONE",    ""),
        "email":    os.getenv("YOUR_EMAIL",    ""),
        "linkedin": os.getenv("YOUR_LINKEDIN", ""),
        "action":   state["action"],
        "subject":  email["subject"],
        "sender":   email["sender"],
        "body":     clipped,
    })

    original_subject = email["subject"]
    reply_subject = (
        original_subject if original_subject.lower().startswith("re:")
        else f"Re: {original_subject}"
    )

    return {
        **state,
        "draft_subject": reply_subject,
        "draft_body":    body,
    }


def skip_node(state: EmailState) -> EmailState:
    """No action needed for this email."""
    return {**state, "draft_body": "", "draft_subject": ""}


# ── Routing ───────────────────────────────────────────────────────────────────
def route_action(state: EmailState) -> Literal["draft_reply", "skip"]:
    if os.getenv("DISABLE_DRAFTS", "false").lower() == "true":
        return "skip"
    if state["action"] in {"DRAFT_FEEDBACK", "DRAFT_CONFIRM", "DRAFT_RESPONSE"}:
        return "draft_reply"
    return "skip"


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_classifier_graph():
    graph = StateGraph(EmailState)

    graph.add_node("classify",      classify_node)
    graph.add_node("decide_action", decide_action_node)
    graph.add_node("draft_reply",   draft_reply_node)
    graph.add_node("skip",          skip_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "decide_action")
    graph.add_conditional_edges(
        "decide_action",
        route_action,
        {"draft_reply": "draft_reply", "skip": "skip"},
    )
    graph.add_edge("draft_reply", END)
    graph.add_edge("skip",        END)

    return graph.compile()


# Compile once at import time
classifier = build_classifier_graph()


def process_email(email: dict) -> EmailState:
    """Run a single email through the full classifier→action→draft pipeline."""
    initial: EmailState = {
        "email":         email,
        "category":      "",
        "action":        "",
        "draft_subject": "",
        "draft_body":    "",
        "reasoning":     "",
    }
    return classifier.invoke(initial)
