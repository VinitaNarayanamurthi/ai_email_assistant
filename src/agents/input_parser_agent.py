import sys
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
"""

USER_PROMPT = """Parse the following email request:

"{raw_input}"

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

    try:
        parsed: dict[str, object] = chain.invoke({"raw_input": raw_input})  # type: ignore[assignment]

        user_profile = state.get("user_profile") or {}
        default_tone = user_profile.get("default_tone")
        if parsed.get("tone_hint") is None and default_tone:
            parsed["tone_hint"] = default_tone

        return {"parsed_context": parsed, "parse_error": None}

    except Exception as e:
        user_profile = state.get("user_profile") or {}
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
