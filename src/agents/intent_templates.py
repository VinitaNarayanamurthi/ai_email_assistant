from typing import TypedDict


class IntentTemplate(TypedDict):
    structure: list[str]
    opener_style: str
    cta_required: bool


INTENT_TEMPLATES: dict[str, IntentTemplate] = {
    "outreach": {
        "structure": ["hook", "value_proposition", "social_proof", "cta"],
        "opener_style": "attention-grabbing",
        "cta_required": True,
    },
    "follow_up": {
        "structure": ["reference_prior", "update_or_ask", "next_step"],
        "opener_style": "referential",
        "cta_required": True,
    },
    "apology": {
        "structure": ["acknowledgment", "explanation_brief", "remedy", "reassurance"],
        "opener_style": "empathetic",
        "cta_required": False,
    },
    "informational": {
        "structure": ["context", "key_points", "implications", "optional_cta"],
        "opener_style": "direct",
        "cta_required": False,
    },
    "internal_update": {
        "structure": ["summary", "details", "action_items"],
        "opener_style": "direct",
        "cta_required": False,
    },
    "request": {
        "structure": ["context", "specific_ask", "rationale", "deadline", "cta"],
        "opener_style": "respectful",
        "cta_required": True,
    },
    "thank_you": {
        "structure": ["gratitude", "specific_what", "forward_looking"],
        "opener_style": "warm",
        "cta_required": False,
    },
    "other": {
        "structure": ["opener", "body", "closing"],
        "opener_style": "neutral",
        "cta_required": False,
    },
}
