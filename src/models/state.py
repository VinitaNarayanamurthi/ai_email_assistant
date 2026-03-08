
import os
from pathlib import Path
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# # Load environment variables from .env file (dynamic path)
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



class ParsedContextDict(TypedDict, total=False):
    recipient_type: str
    recipient_name: str | None
    subject_hint: str
    tone_hint: str | None
    constraints: list[str]
    urgency: bool
    context_notes: str | None


class EmailDraftDict(TypedDict, total=False):
    subject_line: str
    salutation: str
    body_paragraphs: list[str]
    closing: str
    sign_off: str


class ReviewResultDict(TypedDict, total=False):
    grammar_score: float
    tone_alignment_score: float
    coherence_score: float
    structure_complete: bool
    issues: list[str]
    verdict: str


class PriorDraftEntry(TypedDict, total=False):
    date: str
    intent: str
    recipient_type: str
    recipient_name: str | None
    subject_hint: str
    draft_summary: str


class UserProfileDict(TypedDict, total=False):
    user_id: str
    name: str
    company: str
    role: str
    default_tone: str
    sign_off_preference: str
    writing_style_notes: str
    prior_drafts: list[PriorDraftEntry]
    known_contacts: dict[str, str]


class EmailAssistantState(TypedDict, total=False):
    raw_input: str
    ui_tone_selection: str | None
    parsed_context: ParsedContextDict | None
    parse_error: str | None
    intent: str
    intent_confidence: float
    intent_fallback: bool
    tone: str
    tone_directives: list[str]
    tone_sample_ref: str | None
    tone_resolution_log: str
    draft: EmailDraftDict | None
    draft_error: str | None
    personalized_draft: EmailDraftDict | None
    personalization_log: list[str]
    review_result: ReviewResultDict | None
    retry_issues: list[str]
    prior_review_issues: list[str]
    retry_count: int
    active_model: str | None
    user_profile: UserProfileDict
    user_edited_draft: str | None
    final_draft: str | None
    pipeline_status: str | None
    pipeline_warning: str | None
