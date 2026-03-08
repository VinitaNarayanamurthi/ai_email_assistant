from pathlib import Path
from typing import Optional

from src.models.state import EmailAssistantState

SYSTEM_DEFAULT_TONE = "formal"

VALID_TONES = frozenset(
    {"formal", "casual", "assertive", "empathetic", "urgent", "diplomatic"}
)

TONE_SAMPLES_DIR = Path("data/tone_samples")

TONE_DIRECTIVES: dict[str, list[str]] = {
    "formal": [
        "Use precise, professional vocabulary. Avoid slang and colloquialisms.",
        "Do not use contractions (use 'do not' instead of 'don't').",
        "Write in complete sentences with clear structure.",
        "Use a respectful salutation (e.g., 'Dear Mr./Ms. [Name],').",
        "Close with a professional sign-off (e.g., 'Sincerely,' or 'Best regards,').",
        "Maintain a neutral, objective tone throughout.",
    ],
    "casual": [
        "Use a warm, conversational tone.",
        "Contractions are encouraged (e.g., 'I'd', 'we're', 'you've').",
        "Keep sentences short and easy to read.",
        "Use a friendly opener (e.g., 'Hope you're doing well!').",
        "Sign off informally (e.g., 'Cheers,' or 'Thanks,').",
        "It is okay to use light positive language, but avoid exclamation marks in every sentence.",
    ],
    "assertive": [
        "Use direct, active voice throughout.",
        "State the main point or ask in the first or second sentence.",
        "Avoid hedging phrases like 'perhaps', 'might', 'kind of', 'just wanted to'.",
        "Each paragraph should have a single, clear purpose.",
        "End with a specific, actionable call-to-action with a clear deadline if applicable.",
        "Do not apologize unnecessarily or over-explain.",
    ],
    "empathetic": [
        "Lead with acknowledgment of the recipient's feelings or situation.",
        "Use inclusive language ('we understand', 'together we can').",
        "Avoid clinical or detached language.",
        "Validate the recipient's experience before moving to solutions.",
        "Close with a reassuring and supportive statement.",
    ],
    "urgent": [
        "Open with the urgency context immediately — do not bury the lead.",
        "Use bold or clear formatting cues (e.g., 'Action Required:', 'Time-Sensitive:') in the subject line.",
        "Keep the body concise — remove all non-essential information.",
        "State the deadline explicitly (day and time if possible).",
        "Close with a clear, specific next step and contact information.",
    ],
    "diplomatic": [
        "Use balanced, neutral language that does not assign blame.",
        "Acknowledge multiple perspectives before proposing a resolution.",
        "Avoid absolute statements (never, always, impossible).",
        "Use softening phrases where appropriate ('It would be helpful if...', 'We appreciate your patience...').",
        "End on a collaborative, forward-looking note.",
    ],
}

TONE_INTENT_MODIFIERS: dict[tuple[str, str], list[str]] = {
    ("formal", "apology"): [
        "Begin by directly acknowledging the issue without deflecting responsibility.",
        "Do not use passive voice to obscure accountability.",
        "Offer a concrete remedy or next step.",
    ],
    ("assertive", "outreach"): [
        "Open with a specific value proposition — not a generic introduction.",
        "Reference a relevant pain point the recipient likely has.",
        "End with a single, specific call-to-action (not multiple options).",
    ],
    ("casual", "follow_up"): [
        "Reference the prior interaction naturally and briefly.",
        "Keep the tone light — this is a check-in, not a demand.",
        "Make it easy for the recipient to say yes or respond simply.",
    ],
    ("empathetic", "apology"): [
        "Validate any inconvenience caused before offering solutions.",
        "Use first-person accountability language ('I', 'we') — not passive.",
        "Do not rush to solutions before acknowledging impact.",
    ],
    ("urgent", "request"): [
        "State the deadline and consequence of missing it in the first paragraph.",
        "Remove all pleasantries if the urgency is extreme.",
        "Offer to assist or provide whatever is needed to speed resolution.",
    ],
}

TONE_CHANGE_SIGNALS: dict[str, list[str]] = {
    "casual": ["casual", "informal", "friendly", "relaxed", "conversational"],
    "formal": ["formal", "professional", "official", "proper"],
    "assertive": ["assertive", "direct", "confident", "strong"],
    "empathetic": ["empathetic", "warm", "understanding", "compassionate"],
    "urgent": ["urgent", "time-sensitive", "quick", "asap"],
    "diplomatic": ["diplomatic", "balanced", "neutral", "tactful"],
}


def resolve_tone(state: EmailAssistantState) -> tuple[str, str]:
    ui_tone = state.get("ui_tone_selection")
    if ui_tone and ui_tone in VALID_TONES:
        return ui_tone, "ui_selection"

    ctx = state.get("parsed_context") or {}
    prompt_tone = ctx.get("tone_hint")
    if prompt_tone and prompt_tone in VALID_TONES:
        return str(prompt_tone), "prompt_hint"

    profile = state.get("user_profile") or {}
    profile_tone = profile.get("default_tone")
    if profile_tone and profile_tone in VALID_TONES:
        return str(profile_tone), "profile_default"

    return SYSTEM_DEFAULT_TONE, "system_default"


def resolve_tone_sample(tone: str, intent: str) -> Optional[str]:
    user_path = TONE_SAMPLES_DIR / f"{tone}_{intent}_user.txt"
    if user_path.exists():
        return str(user_path)

    specific_path = TONE_SAMPLES_DIR / f"{tone}_{intent}.txt"
    if specific_path.exists():
        return str(specific_path)

    fallback = TONE_SAMPLES_DIR / f"{tone}.txt"
    if fallback.exists():
        return str(fallback)

    return None


def detect_tone_change_in_refinement(raw_input: str) -> Optional[str]:
    lowered = raw_input.lower()
    for tone, signals in TONE_CHANGE_SIGNALS.items():
        if any(signal in lowered for signal in signals):
            return tone
    return None


def tone_stylist_agent(state: EmailAssistantState) -> dict[str, object]:
    raw_input = str(state.get("raw_input") or "")
    prior_tone = state.get("tone")
    intent = str(state.get("intent") or "other")

    tone_override = detect_tone_change_in_refinement(raw_input) if prior_tone else None

    if tone_override:
        directives = list(TONE_DIRECTIVES.get(tone_override, TONE_DIRECTIVES[SYSTEM_DEFAULT_TONE]))
        modifier_key = (tone_override, intent)
        if modifier_key in TONE_INTENT_MODIFIERS:
            directives.extend(TONE_INTENT_MODIFIERS[modifier_key])
        return {
            "tone": tone_override,
            "tone_directives": directives,
            "tone_sample_ref": resolve_tone_sample(tone_override, intent),
            "tone_resolution_log": f"Resolved via: refinement_override (was: {prior_tone})",
        }

    tone, resolution_source = resolve_tone(state)
    directives = list(TONE_DIRECTIVES.get(tone, TONE_DIRECTIVES[SYSTEM_DEFAULT_TONE]))

    modifier_key = (tone, intent)
    if modifier_key in TONE_INTENT_MODIFIERS:
        directives.extend(TONE_INTENT_MODIFIERS[modifier_key])

    tone_sample_ref = resolve_tone_sample(tone, intent)

    return {
        "tone": tone,
        "tone_directives": directives,
        "tone_sample_ref": tone_sample_ref,
        "tone_resolution_log": f"Resolved via: {resolution_source}",
    }
