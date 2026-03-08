import json
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.draft_writer_agent import draft_writer_agent
from src.agents.input_parser_agent import input_parser_agent
from src.agents.intent_detection_agent import intent_detection_agent
from src.agents.personalization_agent import personalization_agent
from src.agents.review_agent import review_validator_agent
from src.agents.router_agent import router_memory_agent
from src.agents.tone_stylist_agent import tone_stylist_agent
from src.models.state import EmailAssistantState, UserProfileDict

PROFILES_PATH = Path("src/memory/user_profiles.json")


def load_user_profile(user_id: str) -> UserProfileDict:
    if not PROFILES_PATH.exists():
        return {}  # type: ignore[return-value]
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles: dict[str, object] = json.load(f)
    profile = profiles.get(user_id) or profiles.get("default") or {}
    return profile  # type: ignore[return-value]


def after_input_parser(state: EmailAssistantState) -> str:
    if state.get("parsed_context") is None:
        return "error_terminal"
    return "intent_detection"


def after_draft_writer(state: EmailAssistantState) -> str:
    if state.get("draft") is None:
        return "error_terminal"
    return "personalization"


def route_after_router(state: EmailAssistantState) -> str:
    status = str(state.get("pipeline_status") or "warning")
    if status in ("success", "warning"):
        return "end"
    return "retry"


def build_graph(use_sqlite: bool = False) -> object:
    graph: StateGraph = StateGraph(EmailAssistantState)

    graph.add_node("input_parser", input_parser_agent)
    graph.add_node("intent_detection", intent_detection_agent)
    graph.add_node("tone_stylist", tone_stylist_agent)
    graph.add_node("draft_writer", draft_writer_agent)
    graph.add_node("personalization", personalization_agent)
    graph.add_node("review_validator", review_validator_agent)
    graph.add_node("router_memory", router_memory_agent)

    graph.set_entry_point("input_parser")

    graph.add_conditional_edges(
        "input_parser",
        after_input_parser,
        {
            "intent_detection": "intent_detection",
            "error_terminal": END,
        },
    )

    graph.add_edge("intent_detection", "tone_stylist")
    graph.add_edge("tone_stylist", "draft_writer")

    graph.add_conditional_edges(
        "draft_writer",
        after_draft_writer,
        {
            "personalization": "personalization",
            "error_terminal": END,
        },
    )

    graph.add_edge("personalization", "review_validator")
    graph.add_edge("review_validator", "router_memory")

    graph.add_conditional_edges(
        "router_memory",
        route_after_router,
        {
            "end": END,
            "retry": "draft_writer",
        },
    )

    if use_sqlite:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-untyped]

            db_path = "data/conversation_memory.db"
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            with SqliteSaver.from_conn_string(db_path) as checkpointer:
                return graph.compile(checkpointer=checkpointer)
        except Exception:
            pass

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_pipeline(
    raw_input: str,
    user_id: str = "default",
    ui_tone_selection: Optional[str] = None,
    thread_id: Optional[str] = None,
    compiled_graph: Optional[object] = None,
) -> dict[str, object]:
    if compiled_graph is None:
        compiled_graph = build_graph()

    profile = load_user_profile(user_id)

    initial_state: EmailAssistantState = {
        "raw_input": raw_input,
        "ui_tone_selection": ui_tone_selection,
        "user_profile": profile,
        "retry_count": 0,
    }

    config: dict[str, object] = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    result: dict[str, object] = compiled_graph.invoke(initial_state, config)  # type: ignore[union-attr]
    return result
