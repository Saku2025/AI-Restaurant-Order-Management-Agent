"""Web API + static frontend for the restaurant order agent.

Run with:  uvicorn server:api --reload
"""
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import BaseModel

from main import MENU, app, new_initial_state, run_config

STATIC_DIR = Path(__file__).parent / "static"

api = FastAPI(title="Restaurant Order Agent")
api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Reply(BaseModel):
    text: str


def menu_payload() -> list[dict]:
    return [
        {"dish": dish, "price": info["price"], "available": info["stock"]}
        for dish, info in MENU.items()
    ]


def run_graph(thread_id: str, graph_input) -> dict:
    """Advance the graph until it pauses or ends; report what happened."""
    config = run_config(thread_id)
    steps = []

    for update in app.stream(graph_input, config, stream_mode="updates"):
        for node, values in update.items():
            if node == "__interrupt__":
                continue
            values = values or {}
            steps.append({
                "node": node,
                "status": values.get("status"),
                "messages": [
                    m.content for m in values.get("messages", [])
                    if isinstance(m, AIMessage)
                ],
            })

    snapshot = app.get_state(config)
    state = snapshot.values
    pending = None
    if snapshot.next and snapshot.tasks and snapshot.tasks[0].interrupts:
        pending = snapshot.tasks[0].interrupts[0].value

    return {
        "session_id": thread_id,
        "steps": steps,
        "waiting_for": pending,
        "done": not snapshot.next,
        "final_result": state.get("final_result") or None,
        "order": {
            "items": state.get("items", []),
            "status": state.get("status"),
        },
        "bill": state.get("bill") or None,
        "payment": state.get("payment") or None,
        "budgets": {
            "order": max(state.get("order_retry_attempts", 0), 0),
            "cook": max(state.get("cook_retry_attempts", 0), 0),
            "serve": max(state.get("serve_retry_attempts", 0), 0),
        },
        "menu": menu_payload(),
    }


@api.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@api.get("/api/menu")
def get_menu():
    return menu_payload()


@api.post("/api/session")
def start_session():
    return run_graph(str(uuid.uuid4()), new_initial_state())


@api.post("/api/session/{session_id}/reply")
def reply(session_id: str, body: Reply):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Message cannot be empty.")

    snapshot = app.get_state(run_config(session_id))
    if not snapshot.values:
        raise HTTPException(404, "Unknown session. Start a new order.")
    if not snapshot.next:
        raise HTTPException(409, "This order is finished. Start a new order.")

    return run_graph(session_id, Command(resume=text))
