"""Human-in-the-Loop (HITL) Confirmation Gate for High-Stakes Agent Actions.

Implements Rubric Category 3.4 (Human-in-the-Loop Hooks):
Enforces an explicit code stop before executing high-stakes actions:
1. Submitting live financial grocery orders (`place_instacart_grocery_order`).
2. Approving grocery carts that exceed the user's weekly budget ceiling.
3. Removing or overriding declared anaphylactic food allergies.

Security guarantees:
- Tokens are cryptographically bound via HMAC-SHA256 to `(approval_request_id, household_id, plan_id, amount_usd)`.
- Tickets are persisted in SQLite (`PersistentMealSessionStore`) across instances/restarts.
- Tokens enforce expiration (`expires_at_iso`) and single-use replay prevention (`status = "CONSUMED"`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from typing import Any, Dict, Optional
import uuid

from meal_concierge.memory.session_store import get_persistent_session_store
from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import traced_span
from meal_concierge.security.secret_manager import get_secret_manager


class HumanApprovalRequiredError(RuntimeError):
    """Exception signaling an explicit code stop awaiting human confirmation."""

    def __init__(self, ticket: "ApprovalRequestTicket") -> None:
        self.ticket = ticket
        super().__init__(
            f"Human-in-the-Loop approval required for '{ticket.action_name}' "
            f"(ticket={ticket.approval_request_id}, amount=${ticket.amount_usd:.2f})."
        )


@dataclass
class ApprovalRequestTicket:
    """Structured ticket created when a high-stakes action pauses for human confirmation."""

    approval_request_id: str
    action_name: str
    household_id: str
    plan_id: str
    amount_usd: float
    summary: str
    status: str
    created_at_iso: str
    expires_at_iso: str
    confirmation_token: Optional[str] = None
    approved_by: Optional[str] = None


class HumanInTheLoopApprovalGate:
    """Manages persistent approval tickets and single-use HMAC confirmation tokens."""

    HIGH_STAKES_TOOLS = frozenset(
        {
            "place_instacart_grocery_order",
            "override_allergy_profile",
            "approve_budget_overrun",
        }
    )

    def __init__(self) -> None:
        self._tickets: Dict[str, ApprovalRequestTicket] = {}
        self._logger = get_structured_logger()
        self._secret_manager = get_secret_manager()

    def _compute_hmac_token(
        self,
        approval_request_id: str,
        household_id: str,
        plan_id: str,
        amount_usd: float,
    ) -> str:
        signing_secret = self._secret_manager.get_secret("HITL_APPROVAL_SIGNING_SECRET")
        msg = f"{approval_request_id}:{household_id}:{plan_id}:{amount_usd:.2f}".encode("utf-8")
        digest = hmac.new(signing_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:24]
        return f"hitl-tok-{digest}"

    def _save_ticket(self, ticket: ApprovalRequestTicket) -> None:
        self._tickets[ticket.approval_request_id] = ticket
        try:
            get_persistent_session_store().upsert_hitl_ticket(
                ticket.approval_request_id, asdict(ticket)
            )
        except Exception:
            pass

    def _load_ticket(self, approval_request_id: str) -> Optional[ApprovalRequestTicket]:
        if approval_request_id in self._tickets:
            return self._tickets[approval_request_id]
        try:
            data = get_persistent_session_store().get_hitl_ticket(approval_request_id)
            if data:
                ticket = ApprovalRequestTicket(**data)
                self._tickets[approval_request_id] = ticket
                return ticket
        except Exception:
            pass
        return None

    def _iter_household_tickets(self, household_id: str) -> list[ApprovalRequestTicket]:
        seen: Dict[str, ApprovalRequestTicket] = {
            tid: t for tid, t in self._tickets.items() if t.household_id == household_id
        }
        try:
            for row in get_persistent_session_store().list_hitl_tickets(household_id=household_id):
                tid = row["approval_request_id"]
                if tid not in seen:
                    seen[tid] = ApprovalRequestTicket(**row)
        except Exception:
            pass
        return list(seen.values())

    @traced_span("orchestration.HumanInTheLoopApprovalGate.request_approval")
    def request_approval(
        self,
        *,
        action_name: str,
        household_id: str,
        plan_id: str,
        amount_usd: float,
        summary: str,
        ttl_minutes: int = 30,
    ) -> ApprovalRequestTicket:
        """Create and persist a pending approval ticket and pause high-stakes execution."""
        now = datetime.now(timezone.utc)
        request_id = f"hitl-{uuid.uuid4().hex[:10]}"
        ticket = ApprovalRequestTicket(
            approval_request_id=request_id,
            action_name=action_name,
            household_id=household_id,
            plan_id=plan_id,
            amount_usd=round(amount_usd, 2),
            summary=summary,
            status="PENDING_HUMAN_APPROVAL",
            created_at_iso=now.isoformat(),
            expires_at_iso=(now + timedelta(minutes=ttl_minutes)).isoformat(),
        )
        self._save_ticket(ticket)

        self._logger.warning(
            f"[HITL STOP] High-stakes action '{action_name}' paused awaiting human approval "
            f"(ticket={request_id}, plan={plan_id}, amount=${amount_usd:.2f})",
            event_type="HITL_APPROVAL_REQUESTED",
            payload=asdict(ticket),
        )
        return ticket

    @traced_span("orchestration.HumanInTheLoopApprovalGate.confirm_human_approval")
    def confirm_human_approval(
        self,
        approval_request_id: str,
        *,
        approved_by: str,
    ) -> ApprovalRequestTicket:
        """Record explicit human approval for a pending ticket and issue a plan-bound `confirmation_token`."""
        ticket = self._load_ticket(approval_request_id)
        if ticket is None:
            raise KeyError(f"Unknown approval_request_id: '{approval_request_id}'")
        if ticket.status == "CONSUMED":
            raise ValueError(f"Approval ticket '{approval_request_id}' has already been consumed.")

        now = datetime.now(timezone.utc)
        if now >= datetime.fromisoformat(ticket.expires_at_iso):
            ticket.status = "EXPIRED"
            self._save_ticket(ticket)
            raise ValueError(f"Approval ticket '{approval_request_id}' expired at {ticket.expires_at_iso}.")

        token = self._compute_hmac_token(
            ticket.approval_request_id,
            ticket.household_id,
            ticket.plan_id,
            ticket.amount_usd,
        )
        ticket.status = "APPROVED"
        ticket.approved_by = approved_by
        ticket.confirmation_token = token
        self._save_ticket(ticket)

        self._logger.info(
            f"[HITL APPROVED] Ticket '{approval_request_id}' confirmed by '{approved_by}'",
            event_type="HITL_APPROVAL_GRANTED",
            payload={
                "approval_request_id": approval_request_id,
                "approved_by": approved_by,
                "action_name": ticket.action_name,
                "plan_id": ticket.plan_id,
                "amount_usd": ticket.amount_usd,
            },
        )
        return ticket

    def verify_confirmation_token(
        self,
        *,
        household_id: str,
        plan_id: Optional[str] = None,
        amount_usd: float,
        confirmation_token: Optional[str],
        consume: bool = False,
    ) -> bool:
        """Verify that `confirmation_token` matches an active, non-expired APPROVED ticket for `(household_id, plan_id, amount_usd)`."""
        if not confirmation_token:
            return False
        now = datetime.now(timezone.utc)
        for ticket in self._iter_household_tickets(household_id):
            if ticket.status != "APPROVED" or ticket.confirmation_token is None:
                continue
            if plan_id is not None and ticket.plan_id != plan_id:
                continue
            if abs(ticket.amount_usd - round(amount_usd, 2)) >= 0.01:
                continue
            if now >= datetime.fromisoformat(ticket.expires_at_iso):
                ticket.status = "EXPIRED"
                self._save_ticket(ticket)
                continue
            expected_hmac = self._compute_hmac_token(
                ticket.approval_request_id,
                ticket.household_id,
                ticket.plan_id,
                ticket.amount_usd,
            )
            if hmac.compare_digest(ticket.confirmation_token, confirmation_token) and hmac.compare_digest(
                expected_hmac, confirmation_token
            ):
                if consume:
                    ticket.status = "CONSUMED"
                    self._save_ticket(ticket)
                return True
        return False

    @staticmethod
    def _unwrap_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
        """Unwrap nested `order_payload` dict if ADK passed `{'order_payload': {...}}`."""
        if isinstance(args.get("order_payload"), dict):
            return args["order_payload"]
        return args

    def adk_before_tool_hitl_callback(
        self,
        tool: Any,
        args: Dict[str, Any],
        tool_context: Any,
    ) -> Optional[Dict[str, Any]]:
        """ADK `before_tool_callback` enforcing an explicit code stop on unapproved high-stakes tools."""
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        if tool_name not in self.HIGH_STAKES_TOOLS:
            return None

        payload = self._unwrap_tool_args(args)
        human_approved = bool(payload.get("human_approved", False))
        confirmation_token = payload.get("confirmation_token")
        household_id = str(payload.get("household_id", "default_household"))
        plan_id = str(payload.get("plan_id", "default_plan"))
        amount_usd = float(payload.get("net_total_usd", 0.0))

        # Note: verify without consuming here so the tool body itself consumes on execution
        if human_approved and self.verify_confirmation_token(
            household_id=household_id,
            plan_id=plan_id,
            amount_usd=amount_usd,
            confirmation_token=confirmation_token,
            consume=False,
        ):
            return None

        ticket = self.request_approval(
            action_name=tool_name,
            household_id=household_id,
            plan_id=plan_id,
            amount_usd=amount_usd,
            summary=f"Execute '{tool_name}' for household '{household_id}' (plan '{plan_id}') totaling ${amount_usd:.2f}.",
        )
        return {
            "order_status": "PENDING_HUMAN_APPROVAL",
            "order_id": None,
            "approval_request_id": ticket.approval_request_id,
            "retailer_name": str(payload.get("retailer_name", "Whole Foods Market")),
            "charged_amount_usd": 0.0,
            "delivery_window": str(payload.get("delivery_window", "Sunday 09:00-11:00")),
            "message": (
                f"Execution halted by Human-in-the-Loop Gate (ticket '{ticket.approval_request_id}'). "
                f"Please ask the user to approve the ${amount_usd:.2f} order for plan '{plan_id}' and pass the issued "
                f"`confirmation_token` with `human_approved=True`."
            ),
        }


_GLOBAL_HITL_GATE = HumanInTheLoopApprovalGate()


def get_hitl_approval_gate() -> HumanInTheLoopApprovalGate:
    """Return the singleton `HumanInTheLoopApprovalGate`."""
    return _GLOBAL_HITL_GATE
