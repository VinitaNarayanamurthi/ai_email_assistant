

import sys

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.models.state import EmailAssistantState

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
result = load_dotenv(str(env_path))

print(f"load_dotenv returned: {result}")

# # Get API keys with validation
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"OPENAI_API_KEY: {'set' if openai_api_key else 'not set'}")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=SecretStr(openai_api_key) if openai_api_key else None,
)


CONFIDENCE_THRESHOLD = 0.65

ALLOWED_INTENTS = frozenset(
    {
        "outreach",
        "follow_up",
        "apology",
        "informational",
        "internal_update",
        "request",
        "thank_you",
        "other",
    }
)

REFINEMENT_SIGNALS = ["shorter", "longer", "tone", "make it", "change it", "less", "more"]

SYSTEM_PROMPT = """You are an expert email intent classifier.

Your only job is to determine the primary purpose of the email being requested.

Supported intents:
- outreach: Cold contact, introduction, partnership or sales proposal
- follow_up: Checking in after a previous email, meeting, or conversation
- apology: Acknowledging a mistake, delay, or failure
- informational: Sharing an update, announcement, or status report
- internal_update: Communication directed at teammates or within an organization
- request: Asking someone for something specific (a document, a meeting, approval)
- thank_you: Expressing gratitude or appreciation
- other: Does not clearly fit any of the above

Rules:
- Choose the SINGLE most dominant intent
- Do not combine intents
- If uncertain between two, pick the one with higher confidence and note it in reasoning
- Return confidence as a decimal between 0.0 and 1.0
"""

USER_PROMPT = """Based on this parsed email context, classify the intent:

Recipient type: {recipient_type}
Subject hint: {subject_hint}
Constraints: {constraints}
Context notes: {context_notes}
{prior_hint}

{format_instructions}

Return a JSON object with fields: intent, confidence, reasoning."""


class IntentResult(BaseModel):
    intent: str = Field(
        description=(
            "Exactly one of: outreach, follow_up, apology, informational, "
            "internal_update, request, thank_you, other"
        )
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="One sentence explaining why this intent was chosen"
    )


def build_intent_chain() -> object:
    parser = JsonOutputParser(pydantic_object=IntentResult)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def get_intent_prior(profile: dict[str, object], recipient_type: str) -> dict[str, float]:
    prior_drafts = profile.get("prior_drafts") or []
    counts: dict[str, int] = {}
    for draft in prior_drafts:  # type: ignore[union-attr]
        if isinstance(draft, dict) and draft.get("recipient_type") == recipient_type:
            intent = str(draft.get("intent") or "other")
            counts[intent] = counts.get(intent, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {}
    return {intent: count / total for intent, count in counts.items()}


def format_prior_hint(prior: dict[str, float]) -> str:
    if not prior:
        return ""
    top = sorted(prior.items(), key=lambda x: x[1], reverse=True)[:2]
    lines = [
        f"  - {intent}: {prob:.0%} of past emails to this recipient type"
        for intent, prob in top
    ]
    return (
        "Historical context (soft signal only — do not override clear intent signals):\n"
        + "\n".join(lines)
    )


def intent_detection_agent(state: EmailAssistantState) -> dict[str, object]:
    raw_input = str(state.get("raw_input") or "")
    prior_intent = state.get("intent")

    if prior_intent and any(s in raw_input.lower() for s in REFINEMENT_SIGNALS):
        return {
            "intent": prior_intent,
            "intent_confidence": 1.0,
            "intent_fallback": False,
        }

    ctx = state.get("parsed_context") or {}
    user_profile = state.get("user_profile") or {}
    recipient_type = str(ctx.get("recipient_type") or "unknown")

    prior = get_intent_prior(dict(user_profile), recipient_type)  # type: ignore[arg-type]
    prior_hint = format_prior_hint(prior)

    chain = build_intent_chain()

    try:
        result: dict[str, object] = chain.invoke(  # type: ignore[assignment]
            {
                "recipient_type": recipient_type,
                "subject_hint": str(ctx.get("subject_hint") or ""),
                "constraints": ", ".join(ctx.get("constraints") or []) or "none",  # type: ignore[arg-type]
                "context_notes": str(ctx.get("context_notes") or "none"),
                "prior_hint": prior_hint,
            }
        )

        intent = str(result.get("intent") or "other")
        confidence = float(result.get("confidence") or 0.0)  # type: ignore[arg-type]
        fallback_used = False

        if confidence < CONFIDENCE_THRESHOLD:
            intent = "other"
            fallback_used = True

        if intent not in ALLOWED_INTENTS:
            intent = "other"
            fallback_used = True

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "intent_fallback": fallback_used,
        }

    except Exception:
        return {
            "intent": "other",
            "intent_confidence": 0.0,
            "intent_fallback": True,
        }
