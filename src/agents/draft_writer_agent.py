import json
import os

from typing import Optional




from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agents.intent_templates import INTENT_TEMPLATES
from src.models.state import EmailAssistantState, EmailDraftDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pathlib import Path
import sys
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

MAX_GENERATION_RETRIES = 2

SURGICAL_EDIT_SIGNALS = [
    "shorter",
    "longer",
    "add",
    "remove",
    "change paragraph",
    "rewrite only",
    "keep the opening",
    "just the closing",
    "p.s.",
    "postscript",
]

SYSTEM_PROMPT = """You are an expert professional email writer.

Your task: write a complete, polished {intent_label} email.

TONE RULES — follow these strictly:
{tone_directives_formatted}

STRUCTURAL REQUIREMENTS for a {intent_label} email:
{structure_requirements}

CONSTRAINTS:
{constraints_formatted}

{retry_instructions}

Output format: Return a JSON object with exactly these fields:
- subject_line (string)
- salutation (string)
- body_paragraphs (list of 2-4 strings, one per paragraph)
- closing (string, 1-2 sentences)
- sign_off (string)

Do not include markdown formatting inside field values. Plain text only."""

USER_PROMPT = """Write the email with these details:

Recipient type: {recipient_type}
Recipient name: {recipient_name}
Subject / topic: {subject_hint}
Additional context: {context_notes}

{few_shot_section}

Now write the email:"""

SURGICAL_EDIT_SYSTEM_PROMPT = """You are editing a specific part of an existing email draft.

Apply ONLY the changes described in the edit instruction.
Preserve all other parts of the draft exactly as they are.
Return the complete updated draft as the same JSON structure.

Existing draft:
{existing_draft}"""

SURGICAL_EDIT_USER_PROMPT = 'Edit instruction: "{edit_instruction}"\n\nReturn the updated draft JSON.'


class EmailDraftSchema(BaseModel):
    subject_line: str = Field(
        description="A clear, specific subject line for the email (5-12 words)"
    )
    salutation: str = Field(
        description="Opening greeting (e.g., 'Dear Mr. Smith,' or 'Hi Sarah,')"
    )
    body_paragraphs: list[str] = Field(
        description=(
            "List of 2-4 body paragraphs. Each paragraph is a single string. "
            "Do not include salutation or sign-off here."
        )
    )
    closing: str = Field(
        description="A single closing sentence or call-to-action (1-2 sentences)"
    )
    sign_off: str = Field(
        description="Sign-off phrase only (e.g., 'Best regards,' or 'Sincerely,'). Do not include name."
    )


def format_tone_directives(directives: list[str]) -> str:
    return "\n".join(f"{i + 1}. {d}" for i, d in enumerate(directives))


def format_structure_requirements(intent: str) -> str:
    template = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["other"])
    structure = template["structure"]
    lines = [f"- Include a section for: {section.replace('_', ' ')}" for section in structure]
    if template.get("cta_required"):
        lines.append("- A clear call-to-action is REQUIRED.")
    return "\n".join(lines)


def format_constraints(ctx: dict[str, object], retry_issues: list[str]) -> str:
    parts: list[str] = []
    constraints = ctx.get("constraints")
    if constraints and isinstance(constraints, list):
        parts.append("User constraints: " + ", ".join(str(c) for c in constraints))
    if ctx.get("urgency"):
        parts.append("This is an urgent email — keep it concise and action-oriented.")
    if not parts:
        parts.append("No special constraints.")
    return "\n".join(parts)


def format_retry_instructions(retry_issues: list[str], retry_count: int) -> str:
    if not retry_issues or retry_count == 0:
        return ""
    issues_str = "\n".join(f"- {issue}" for issue in retry_issues)
    return (
        f"IMPORTANT — This is retry attempt {retry_count}.\n"
        f"The previous draft was rejected for these reasons:\n{issues_str}\n"
        "Fix all of these issues in this new draft."
    )


def load_tone_sample(tone_sample_ref: Optional[str]) -> str:
    if not tone_sample_ref:
        return ""
    path = Path(tone_sample_ref)
    if path.exists():
        sample_text = path.read_text(encoding="utf-8")
        return (
            f'Reference example (match this tone and style, NOT the content):\n"""\n{sample_text}\n"""'
        )
    return ""


def is_surgical_edit(raw_input: str, prior_draft: Optional[EmailDraftDict]) -> bool:
    if not prior_draft:
        return False
    lowered = raw_input.lower()
    return any(signal in lowered for signal in SURGICAL_EDIT_SIGNALS)


def _build_surgical_edit_chain(model: str) -> object:
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SURGICAL_EDIT_SYSTEM_PROMPT),
            ("human", SURGICAL_EDIT_USER_PROMPT),
        ]
    )
    return prompt | llm | JsonOutputParser()


def _build_draft_chain(model: str, temperature: float) -> object:
    
    parser = JsonOutputParser(pydantic_object=EmailDraftSchema)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    return prompt | llm | parser


def apply_surgical_edit(state: EmailAssistantState) -> dict[str, object]:
    prior_draft = state.get("draft") or state.get("personalized_draft")
    raw_input = str(state.get("raw_input") or "")
    active_model = str(state.get("active_model") or "gpt-4o")

    chain = _build_surgical_edit_chain(active_model)

    try:
        updated_draft = chain.invoke(  # type: ignore[attr-defined]
            {
                "existing_draft": json.dumps(prior_draft, indent=2),
                "edit_instruction": raw_input,
            }
        )
        return {"draft": updated_draft, "draft_error": None}
    except Exception as e:
        return {"draft": prior_draft, "draft_error": f"Surgical edit failed: {str(e)}"}


def draft_writer_agent(state: EmailAssistantState) -> dict[str, object]:
    raw_input = str(state.get("raw_input") or "")
    prior_draft = state.get("draft") or state.get("personalized_draft")

    if is_surgical_edit(raw_input, prior_draft):
        return apply_surgical_edit(state)

    ctx = state.get("parsed_context") or {}
    intent = str(state.get("intent") or "other")
    tone_directives: list[str] = list(state.get("tone_directives") or [])  # type: ignore[arg-type]
    tone_sample_ref = state.get("tone_sample_ref")
    retry_issues: list[str] = list(state.get("retry_issues") or [])  # type: ignore[arg-type]
    retry_count = int(state.get("retry_count") or 0)
    active_model = str(state.get("active_model") or "gpt-4o")

    tone_directives_formatted = format_tone_directives(tone_directives)
    structure_requirements = format_structure_requirements(intent)
    constraints_formatted = format_constraints(dict(ctx), retry_issues)  # type: ignore[arg-type]
    retry_instructions = format_retry_instructions(retry_issues, retry_count)
    few_shot_section = load_tone_sample(tone_sample_ref)

    temperature = 0.4 + (0.1 * min(retry_count, 2))
    chain = _build_draft_chain(active_model, temperature)

    for attempt in range(MAX_GENERATION_RETRIES):
        try:
            draft = chain.invoke(  # type: ignore[attr-defined]
                {
                    "intent_label": intent.replace("_", " "),
                    "tone_directives_formatted": tone_directives_formatted,
                    "structure_requirements": structure_requirements,
                    "constraints_formatted": constraints_formatted,
                    "retry_instructions": retry_instructions,
                    "recipient_type": str(ctx.get("recipient_type") or "recipient"),
                    "recipient_name": str(ctx.get("recipient_name") or "[Name]"),
                    "subject_hint": str(ctx.get("subject_hint") or ""),
                    "context_notes": str(ctx.get("context_notes") or "None provided."),
                    "few_shot_section": few_shot_section,
                }
            )
            return {"draft": draft, "draft_error": None}

        except Exception as e:
            if attempt == MAX_GENERATION_RETRIES - 1:
                return {
                    "draft": None,
                    "draft_error": f"Draft generation failed after {MAX_GENERATION_RETRIES} attempts: {str(e)}",
                }

    return {
        "draft": None,
        "draft_error": "Draft generation failed: unexpected exit from retry loop.",
    }
