"""Interactive Web UI & REST API Server for testing the ChefAura Meal Planning Concierge Agent.

Provides:
- `GET  /`                  : Full interactive ChefAura Meal Planning Concierge UI Studio
- `GET  /api/status`        : Agent hierarchy, model routing, pantry stock, and GCP deployment status
- `POST /api/invoke`        : Invokes `MealConciergeRunner.run_weekly_meal_plan` (or remote Vertex AI
                              Agent Engine in `agiljohn-cert-oct2026` / `us-east1`)
- `POST /api/approve_order` : Confirms a Human-in-the-Loop (HITL) ticket (`HMAC-SHA256` token) and
                              executes `place_instacart_grocery_order` (with replay-attack testing)
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import traceback
from typing import Any, Dict

from deploy_agent_platform import (
    DEFAULT_DISPLAY_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    METADATA_PATH,
    query_deployed_agent,
)
from meal_concierge.agent import MealConciergeRunner, build_meal_concierge_agent_hierarchy
from meal_concierge.observability.intent_outcome_auditor import get_global_auditor
from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import get_tracer
from meal_concierge.orchestration.hitl_hooks import get_hitl_approval_gate
from meal_concierge.tools import (
    check_pantry_inventory_levels,
    place_instacart_grocery_order,
    reset_pantry_inventory_store,
)

_RUNNER = MealConciergeRunner()
HTML_UI_PATH = Path(__file__).resolve().parent / "ui_studio.html"


def get_system_status() -> Dict[str, Any]:
    """Return live status of the ADK agent hierarchy, pantry store, and GCP deployment."""
    coordinator = build_meal_concierge_agent_hierarchy()
    deployment_meta: Dict[str, Any] = {
        "project_id": DEFAULT_PROJECT_ID,
        "location": DEFAULT_LOCATION,
        "display_name": DEFAULT_DISPLAY_NAME,
        "deployed": False,
        "pantheon_agent_engine_url": (
            f"https://pantheon.corp.google.com/vertex-ai/agents/agent-engines"
            f"?project={DEFAULT_PROJECT_ID}"
        ),
        "pantheon_agent_registry_url": (
            f"https://pantheon.corp.google.com/agent-management/agent-registry/agents"
            f"?project={DEFAULT_PROJECT_ID}"
        ),
    }
    if METADATA_PATH.exists():
        try:
            saved = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            deployment_meta.update(saved)
            deployment_meta["deployed"] = bool(saved.get("remote_agent_runtime_id"))
        except Exception:
            pass

    pantry = check_pantry_inventory_levels(household_id="hh_demo_ui", include_expiring_within_days=5)
    return {
        "status": "ok",
        "coordinator_name": coordinator.name,
        "coordinator_model": coordinator.model,
        "deployment": deployment_meta,
        "pantry_snapshot": pantry,
    }


def handle_invoke_agent(body: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the Meal Planning Concierge Agent and return the full response + telemetry."""
    get_tracer().exporter.clear()
    get_global_auditor().clear()
    get_structured_logger().clear_buffer()

    if body.get("reset_pantry", True):
        reset_pantry_inventory_store()

    user_prompt = str(
        body.get(
            "user_prompt",
            "Plan my 7-day Mediterranean meals with high protein, zero peanuts/shellfish, and use my expiring spinach and quinoa.",
        )
    )
    dietary_profile = body.get("dietary_profile") or {
        "user_id": "agiljohn",
        "age_years": int(body.get("age_years", 34)),
        "sex_at_birth": str(body.get("sex_at_birth", "male")),
        "weight_kg": float(body.get("weight_kg", 76.0)),
        "height_cm": float(body.get("height_cm", 178.0)),
        "activity_level": str(body.get("activity_level", "moderately_active")),
        "dietary_goal": str(body.get("dietary_goal", "lean_bulk")),
        "dietary_restriction": str(body.get("dietary_restriction", "mediterranean")),
        "allergies": list(body.get("allergies", ["peanuts", "shellfish"])),
        "weekly_budget_usd": float(body.get("weekly_budget_usd", 150.0)),
        "prefer_ingredients": list(body.get("prefer_ingredients", ["quinoa", "spinach", "salmon"])),
    }
    household_id = str(body.get("household_id", "hh_agiljohn"))
    session_id = str(body.get("session_id", "sess_ui_interactive"))
    execution_mode = str(body.get("execution_mode", "local_adk_runner"))

    result = _RUNNER.run_weekly_meal_plan(
        user_prompt=user_prompt,
        dietary_profile=dietary_profile,
        household_id=household_id,
        session_id=session_id,
        human_approved_order=False,
    )

    remote_events = None
    if execution_mode == "vertex_agent_engine":
        try:
            remote_res = query_deployed_agent(
                prompt=f"{user_prompt} | Profile: {json.dumps(dietary_profile)}",
                user_id=dietary_profile.get("user_id", "agiljohn"),
            )
            remote_events = remote_res.get("events", [])
        except Exception as exc:
            remote_events = [{"remote_notice": str(exc)}]

    spans = [
        {
            "name": s.name,
            "trace_id": s.trace_id,
            "span_id": s.span_id,
            "parent_span_id": s.parent_span_id,
            "duration_ms": s.duration_ms,
            "status": s.status,
            "traceparent": s.attributes.get("w3c.traceparent", ""),
        }
        for s in get_tracer().exporter.get_finished_spans()
    ]
    audit_trail = get_global_auditor().get_paired_audit_trail()
    logs = get_structured_logger().get_buffered_records()[-15:]

    return {
        "agent_response": result,
        "remote_agent_engine_events": remote_events,
        "telemetry": {
            "span_count": len(spans),
            "spans": spans,
            "intent_outcome_pairs": audit_trail,
            "structured_logs": logs,
        },
    }


def handle_approve_order(body: Dict[str, Any]) -> Dict[str, Any]:
    """Approve an HITL ticket (or test replay protection) and call `place_instacart_grocery_order`."""
    ticket_id = str(body.get("approval_request_id", ""))
    household_id = str(body.get("household_id", "hh_agiljohn"))
    plan_id = str(body.get("plan_id", ""))
    net_total_usd = float(body.get("net_total_usd", 0.0))
    weekly_budget_usd = float(body.get("weekly_budget_usd", 150.0))
    retailer_name = str(body.get("retailer_name", "Whole Foods Market via Instacart"))
    delivery_window = str(body.get("delivery_window", "Sunday 09:00-11:00"))
    provided_token = body.get("confirmation_token")

    gate = get_hitl_approval_gate()
    if not provided_token:
        ticket = gate.confirm_human_approval(ticket_id, approved_by="agiljohn")
        token_to_use = ticket.confirmation_token
    else:
        token_to_use = str(provided_token)

    order_result = place_instacart_grocery_order(
        household_id=household_id,
        plan_id=plan_id,
        retailer_name=retailer_name,
        delivery_window=delivery_window,
        net_total_usd=net_total_usd,
        weekly_budget_usd=weekly_budget_usd,
        human_approved=True,
        confirmation_token=token_to_use,
    )
    return {
        "order_result": order_result,
        "confirmation_token_used": token_to_use,
    }


class MealConciergeUIHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the interactive UI Studio and JSON API endpoints with CORS."""

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            if HTML_UI_PATH.exists():
                html_bytes = HTML_UI_PATH.read_bytes()
            else:
                html_bytes = b"<h1>ChefAura UI Studio</h1>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        if self.path.startswith("/api/status"):
            self._send_json(200, get_system_status())
            return

        self._send_json(404, {"error": f"Unknown endpoint {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            body = {}

        try:
            if self.path == "/api/invoke":
                self._send_json(200, handle_invoke_agent(body))
                return
            if self.path == "/api/approve_order":
                self._send_json(200, handle_approve_order(body))
                return
            self._send_json(404, {"error": f"Unknown POST endpoint {self.path}"})
        except Exception as exc:
            self._send_json(
                500,
                {
                    "status": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ChefAura Meal Planning Concierge UI Server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8090")),
        help="Bind port (default: 8090)",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MealConciergeUIHandler)
    print(f"ChefAura Meal Planning Concierge UI Studio listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
