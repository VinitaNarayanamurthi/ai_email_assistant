import sys
import json
from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.models.state import EmailAssistantState, ParsedContextDict
import os
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


REFINEMENT_SIGNALS = [
    "make it",
    "change it",
    "shorter",
    "longer",
    "more formal",
    "less formal",
    "add",
    "remove",
    "different tone",
    "update",
]

SYSTEM_PROMPT = """You are an expert at parsing natural language instructions for email composition.

Extract structured information from the user's email request. Be conservative:
- Only populate fields that are clearly stated or strongly implied
- Use null for any field not mentioned
- Do not infer or hallucinate details the user did not provide

Recipient types: client, colleague, manager, partner, vendor, other
Tone hints (only if user explicitly states): formal, casual, assertive

For context_notes — capture any situational background the draft writer needs:
- Prior decisions: "request was previously declined", "they said no", "she approved"
- Relationship history: "met at conference last week", "ongoing project since Q3"
- Specific goals: "need approval by Friday", "want to persuade", "address their objection"
- Any conversational context the user references (e.g. "He said no") that informs the email's angle
"""

USER_PROMPT = """Parse the following email request:

"{raw_input}"

{prior_context_section}

{format_instructions}

Return a JSON object matching the schema exactly. Do not add extra fields."""

REFINEMENT_SYSTEM_PROMPT = """You are updating a structured email request context based on a user's refinement command.
Only change the fields explicitly mentioned in the refinement.
Return the full updated JSON context with all original fields preserved unless changed.
Original context: {prior_context}"""

REFINEMENT_USER_PROMPT = 'Refinement command: "{refinement}"\n\nReturn the updated JSON context.'


class ParsedContext(BaseModel):
    recipient_type: str = Field(
        description="Type of recipient: client, colleague, manager, partner, vendor, other"
    )
    recipient_name: Optional[str] = Field(
        default=None,
        description="Specific name of recipient if mentioned, else null",
    )
    subject_hint: str = Field(
        description="The core topic or subject matter of the email in 3-8 words"
    )
    tone_hint: Optional[str] = Field(
        default=None,
        description="Tone explicitly requested by user: formal, casual, assertive, or null",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Any explicit constraints: concise, apologetic, urgent, avoid-blame, etc.",
    )
    urgency: bool = Field(
        default=False,
        description="True if user indicated time-sensitivity (ASAP, urgent, today, etc.)",
    )
    context_notes: Optional[str] = Field(
        default=None,
        description="Any additional context that does not fit above fields",
    )


def build_parser_chain() -> object:
    parser = JsonOutputParser(pydantic_object=ParsedContext)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def _format_prior_context_section(prior: ParsedContextDict) -> str:
    parts: list[str] = [
        "Conversation context from the previous email in this session "
        "(carry over any details still relevant — do NOT override what the user explicitly states in the new request):"
    ]
    recipient_type = prior.get("recipient_type")
    recipient_name = prior.get("recipient_name")
    subject_hint = prior.get("subject_hint")
    tone_hint = prior.get("tone_hint")
    if recipient_type:
        parts.append(f"  - Recipient type: {recipient_type}")
    if recipient_name:
        parts.append(f"  - Recipient name: {recipient_name}")
    if subject_hint:
        parts.append(f"  - Previous subject: {subject_hint}")
    if tone_hint:
        parts.append(f"  - Established tone: {tone_hint}")
    return "\n".join(parts)


def is_refinement_input(raw_input: str) -> bool:
    lowered = raw_input.lower().strip()
    return any(signal in lowered for signal in REFINEMENT_SIGNALS) and len(lowered) < 100


def _build_refinement_chain() -> object:
   
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REFINEMENT_SYSTEM_PROMPT),
            ("human", REFINEMENT_USER_PROMPT),
        ]
    )
    return prompt | llm | JsonOutputParser()


def apply_refinement_to_context(
    refinement: str, prior_context: ParsedContextDict
) -> dict[str, object]:
    chain = _build_refinement_chain()
    try:
        updated = chain.invoke(  # type: ignore[attr-defined]
            {"prior_context": str(prior_context), "refinement": refinement}
        )
        return {"parsed_context": updated, "parse_error": None}
    except Exception:
        return {"parsed_context": prior_context, "parse_error": None}


def validate_parsed_context(parsed: dict[str, object]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not parsed.get("subject_hint"):
        issues.append("Could not determine email subject. Please be more specific.")
    if not parsed.get("recipient_type"):
        issues.append("Could not determine who this email is for.")
    return len(issues) == 0, issues


def input_parser_agent(state: EmailAssistantState) -> dict[str, object]:
    raw_input = str(state.get("raw_input") or "").strip()

    if not raw_input:
        return {
            "parsed_context": None,
            "parse_error": "Input is empty. Please describe the email you want to write.",
        }

    prior_context = state.get("parsed_context")
    if prior_context and is_refinement_input(raw_input):
        return apply_refinement_to_context(raw_input, prior_context)

    chain = build_parser_chain()

    prior_context_section = (
        _format_prior_context_section(prior_context)
        if prior_context
        else ""
    )

    try:
        parsed: dict[str, object] = chain.invoke(  # type: ignore[assignment]
            {"raw_input": raw_input, "prior_context_section": prior_context_section}
        )

        user_profile = state.get("user_profile") or {}
        default_tone = user_profile.get("default_tone")
        if parsed.get("tone_hint") is None and default_tone:
            parsed["tone_hint"] = default_tone

        return {"parsed_context": parsed, "parse_error": None}

    except Exception as e:
        user_profile = state.get("user_profile") or {}  # noqa: F841 (already read above)
        fallback: ParsedContextDict = {
            "recipient_type": "recipient",
            "recipient_name": None,
            "subject_hint": raw_input[:60],
            "tone_hint": str(user_profile.get("default_tone") or "formal"),
            "constraints": [],
            "urgency": False,
            "context_notes": raw_input,
        }
        return {
            "parsed_context": fallback,
            "parse_error": f"Parsing degraded: {str(e)}",
        }


def main() -> None:
    """Run a quick manual test for the input parser agent."""
    prompt = (
       
        "Write a formal apology email to client Sarah for a delayed shipment."
    )

    test_state: EmailAssistantState = {
        "raw_input": prompt,
        "user_profile": {"default_tone": "formal"},
    }

    output = input_parser_agent(test_state)
    parsed_context_raw = output.get("parsed_context")
    parsed_context: dict[str, object] = {}
    if isinstance(parsed_context_raw, dict):
        parsed_context = {str(k): v for k, v in parsed_context_raw.items()}

    # Some model responses may use alternate keys; normalize for test validation.
    if not parsed_context.get("recipient_name") and parsed_context.get("recipient"):
        parsed_context["recipient_name"] = parsed_context.get("recipient")
    if not parsed_context.get("subject_hint") and parsed_context.get("subject"):
        parsed_context["subject_hint"] = parsed_context.get("subject")
    if not parsed_context.get("subject_hint") and parsed_context.get("body"):
        parsed_context["subject_hint"] = parsed_context.get("body")
    if not parsed_context.get("tone_hint") and parsed_context.get("tone"):
        parsed_context["tone_hint"] = parsed_context.get("tone")

    is_valid, issues = validate_parsed_context(parsed_context)

    print("=== INPUT PARSER TEST ===")
    print(f"Prompt: {prompt}")
    print(f"Parse error: {output.get('parse_error')}")
    print(f"Valid parsed context: {is_valid}")
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")

    print("\nParsed context JSON:")
    print(json.dumps(parsed_context, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
