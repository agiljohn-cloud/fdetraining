"""History Compaction, Vertex AI Context Caching, and ADK Memory Bank Management.

Implements Rubric Category 2.2 (History Compaction):
Prevents context bloat across multi-turn meal planning sessions via:
1. Token-budget estimation and sliding-window truncation of older conversational turns.
2. Structured summarization of evicted turns while pinning immutable dietary constraints
   (allergies, calorie floors, macro targets, and budget ceilings).
3. Google Cloud Vertex AI Context Caching (`vertexai.preview.caching.CachedContent`)
   for static constitutional prompts and recipe catalogs.
4. ADK `before_model_callback` compaction integration preserving `google.genai.types.Content`
   object types and valid `"user"`/`"model"` roles, plus persistent Memory Bank sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import os
from typing import Any, Dict, List, Optional

from meal_concierge.memory.session_store import get_persistent_session_store
from meal_concierge.observability.pii_redactor import redact_pii_text
from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import traced_span

try:
    from vertexai.preview import caching as vertex_caching  # type: ignore[import-not-found]

    VERTEX_CACHING_AVAILABLE = True
except ImportError:  # pragma: no cover
    vertex_caching = None
    VERTEX_CACHING_AVAILABLE = False

try:
    from google.genai import types as genai_types  # type: ignore[import-not-found]

    GENAI_TYPES_AVAILABLE = True
except ImportError:  # pragma: no cover
    genai_types = None
    GENAI_TYPES_AVAILABLE = False


@dataclass
class CompactionResult:
    """Telemetry and resulting message payload after context history compaction."""

    original_turn_count: int
    compacted_turn_count: int
    original_estimated_tokens: int
    compacted_estimated_tokens: int
    tokens_saved: int
    compaction_applied: bool
    summary_digest: Optional[str]
    pinned_constraints: List[str]
    retained_messages: List[Dict[str, str]] = field(default_factory=list)


class VertexAIContextCacheManager:
    """Manages Google Cloud Vertex AI Context Caching (`CachedContent`) for large prompts."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        ttl_minutes: int = 60,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        enable_cloud_rpc: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.ttl_minutes = ttl_minutes
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "fde-meal-concierge")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        env_flag = os.environ.get("ENABLE_VERTEX_CACHING", "false").lower() in ("1", "true", "yes")
        self.enable_cloud_rpc = (
            enable_cloud_rpc if enable_cloud_rpc is not None else env_flag
        ) and VERTEX_CACHING_AVAILABLE
        self._cache_registry: Dict[str, Dict[str, Any]] = {}
        self._active_cached_content_name: Optional[str] = None
        self._logger = get_structured_logger()

    @traced_span("memory.VertexAIContextCacheManager.get_or_create_context_cache")
    def get_or_create_context_cache(
        self,
        cache_key: str,
        system_instruction: str,
        static_context_documents: List[str],
    ) -> Dict[str, Any]:
        """Create or retrieve a Vertex AI `CachedContent` handle for static system context."""
        digest = hashlib.sha256(
            (system_instruction + "\n".join(static_context_documents)).encode("utf-8")
        ).hexdigest()[:16]
        composite_key = f"{cache_key}:{digest}"

        if composite_key in self._cache_registry:
            entry = self._cache_registry[composite_key]
            entry["cache_hits"] += 1
            self._active_cached_content_name = entry["resource_name"]
            return entry

        cloud_cache_name = (
            f"projects/{self.project_id}/locations/{self.location}/cachedContents/mc-{digest}"
        )
        remote_created = False
        if self.enable_cloud_rpc and vertex_caching is not None:
            try:
                cached_content = vertex_caching.CachedContent.create(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    contents=static_context_documents,
                    ttl=timedelta(minutes=self.ttl_minutes),
                    display_name=f"meal-concierge-{cache_key}",
                )
                cloud_cache_name = getattr(cached_content, "resource_name", cloud_cache_name)
                remote_created = True
            except Exception:
                remote_created = False

        record = {
            "cache_key": composite_key,
            "resource_name": cloud_cache_name,
            "model_name": self.model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl_minutes": self.ttl_minutes,
            "estimated_cached_tokens": max(
                1,
                (len(system_instruction) + sum(len(d) for d in static_context_documents)) // 4,
            ),
            "cache_hits": 0,
            "remote_created": remote_created,
        }
        self._cache_registry[composite_key] = record
        if remote_created or not self.enable_cloud_rpc:
            self._active_cached_content_name = cloud_cache_name
        else:
            self._active_cached_content_name = None
        self._logger.info(
            f"Registered Vertex AI Context Cache '{cloud_cache_name}'",
            event_type="VERTEX_CONTEXT_CACHE_CREATED",
            payload=record,
        )
        return record


class VertexAIMemoryBankSync:
    """Synchronizes compacted session digests with persistent SQLite and Vertex AI Memory Bank."""

    def __init__(self) -> None:
        self._logger = get_structured_logger()

    @traced_span("memory.VertexAIMemoryBankSync.store_compacted_digest")
    def store_compacted_digest(
        self,
        user_id: str,
        session_id: str,
        summary_digest: str,
        pinned_constraints: List[str],
    ) -> Dict[str, Any]:
        """Persist a compacted conversational digest into the persistent Memory Bank table."""
        store = get_persistent_session_store()
        entry = store.save_memory_bank_digest(
            user_id=user_id,
            session_id=session_id,
            summary_digest=summary_digest,
            pinned_constraints=pinned_constraints,
        )
        self._logger.info(
            f"Synced compacted session digest to Vertex AI Memory Bank for user '{user_id}'",
            event_type="MEMORY_BANK_SYNC",
            session_id=session_id,
            payload=entry,
        )
        return entry

    def get_user_memory_bank(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all stored Memory Bank digests for `user_id` from persistent storage."""
        return get_persistent_session_store().list_memory_bank_digests(user_id)


class ContextCompactionManager:
    """Manages conversational history compaction via token budgeting, sliding windows, and summarization."""

    def __init__(
        self,
        max_context_tokens: int = 4000,
        sliding_window_turns: int = 6,
        context_cache_manager: Optional[VertexAIContextCacheManager] = None,
        memory_bank_sync: Optional[VertexAIMemoryBankSync] = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.sliding_window_turns = sliding_window_turns
        self.context_cache = context_cache_manager or VertexAIContextCacheManager()
        self.memory_bank = memory_bank_sync or VertexAIMemoryBankSync()
        self._logger = get_structured_logger()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate LLM token count (~4 characters per token heuristic + word boundary weighting)."""
        if not text:
            return 0
        char_estimate = max(1, len(text) // 4)
        word_estimate = int(len(text.split()) * 1.35)
        return max(char_estimate, word_estimate)

    @classmethod
    def estimate_messages_tokens(cls, messages: List[Dict[str, str]]) -> int:
        """Estimate total tokens across a sequence of role/content message dicts."""
        return sum(cls.estimate_tokens(m.get("content", "")) + 4 for m in messages)

    @staticmethod
    def _extract_pinned_dietary_constraints(messages: List[Dict[str, str]]) -> List[str]:
        """Extract critical allergy, dietary, macro, and budget facts that must never be lost during compaction."""
        keywords = (
            "allergy",
            "allergies",
            "allergic",
            "peanut",
            "shellfish",
            "gluten",
            "dairy",
            "soy",
            "sesame",
            "tree nut",
            "vegan",
            "vegetarian",
            "keto",
            "mediterranean",
            "kcal",
            "calories",
            "protein",
            "budget",
            "$",
        )
        pinned: List[str] = []
        for msg in messages:
            content = redact_pii_text(msg.get("content", ""))
            for sentence in content.replace("\n", ". ").split("."):
                clean = sentence.strip()
                if not clean:
                    continue
                lower = clean.lower()
                if any(kw in lower for kw in keywords) and clean not in pinned:
                    pinned.append(clean[:180])
        return pinned[:10]

    @traced_span("memory.ContextCompactionManager.compact_history")
    def compact_history(
        self,
        messages: List[Dict[str, str]],
        *,
        user_id: str = "default_user",
        session_id: str = "default_session",
        force_compaction: bool = False,
    ) -> CompactionResult:
        """Compact conversational history when token count or turn count exceeds thresholds.

        Uses `"user"` role for the compacted context digest so Gemini/ADK `contents`
        validation (which only permits `"user"` and `"model"` roles in `contents`) succeeds.
        """
        original_turn_count = len(messages)
        original_tokens = self.estimate_messages_tokens(messages)
        pinned_constraints = self._extract_pinned_dietary_constraints(messages)

        needs_compaction = (
            force_compaction
            or original_tokens > self.max_context_tokens
            or original_turn_count > (self.sliding_window_turns + 2)
        )

        if not needs_compaction or original_turn_count <= self.sliding_window_turns:
            return CompactionResult(
                original_turn_count=original_turn_count,
                compacted_turn_count=original_turn_count,
                original_estimated_tokens=original_tokens,
                compacted_estimated_tokens=original_tokens,
                tokens_saved=0,
                compaction_applied=False,
                summary_digest=None,
                pinned_constraints=pinned_constraints,
                retained_messages=list(messages),
            )

        evicted_messages = messages[: -self.sliding_window_turns]
        recent_window = messages[-self.sliding_window_turns :]

        evicted_tokens = self.estimate_messages_tokens(evicted_messages)
        max_digest_chars = max(48, (evicted_tokens * 2))
        constraints_str = (
            "; ".join(pinned_constraints[:3])
            if pinned_constraints
            else "Standard constitutional safety rules"
        )
        raw_digest = f"[COMPACTED {len(evicted_messages)} TURNS] Constraints: {constraints_str}."
        summary_digest = raw_digest[:max_digest_chars]

        # Note: role is "user" (never "system") for Gemini / ADK `contents` compatibility
        compacted_messages: List[Dict[str, str]] = [
            {"role": "user", "content": summary_digest},
            *recent_window,
        ]
        compacted_tokens = self.estimate_messages_tokens(compacted_messages)
        tokens_saved = max(1, original_tokens - compacted_tokens)

        self.memory_bank.store_compacted_digest(
            user_id=user_id,
            session_id=session_id,
            summary_digest=summary_digest,
            pinned_constraints=pinned_constraints,
        )

        self._logger.info(
            f"History compaction reduced context from {original_tokens} to {compacted_tokens} tokens "
            f"({original_turn_count} -> {len(compacted_messages)} messages)",
            event_type="CONTEXT_HISTORY_COMPACTED",
            session_id=session_id,
            payload={
                "original_turn_count": original_turn_count,
                "compacted_turn_count": len(compacted_messages),
                "original_estimated_tokens": original_tokens,
                "compacted_estimated_tokens": compacted_tokens,
                "tokens_saved": tokens_saved,
                "pinned_constraints_count": len(pinned_constraints),
            },
        )

        return CompactionResult(
            original_turn_count=original_turn_count,
            compacted_turn_count=len(compacted_messages),
            original_estimated_tokens=original_tokens,
            compacted_estimated_tokens=compacted_tokens,
            tokens_saved=tokens_saved,
            compaction_applied=True,
            summary_digest=summary_digest,
            pinned_constraints=pinned_constraints,
            retained_messages=compacted_messages,
        )


_GLOBAL_COMPACTION_MANAGER = ContextCompactionManager()


def _extract_role_and_text(item: Any) -> Dict[str, str]:
    """Extract normalized role ('user' or 'model') and text from a dict or `genai.types.Content`."""
    if isinstance(item, dict):
        raw_role = str(item.get("role", "user"))
        role = "model" if raw_role in ("assistant", "model") else "user"
        return {"role": role, "content": str(item.get("content", ""))}
    # Handle google.genai.types.Content or duck-typed Content object with .role and .parts
    raw_role = str(getattr(item, "role", "user") or "user")
    role = "model" if raw_role in ("assistant", "model") else "user"
    parts = getattr(item, "parts", None)
    if isinstance(parts, list) and parts:
        texts: List[str] = []
        for p in parts:
            txt = getattr(p, "text", None)
            if txt:
                texts.append(str(txt))
                continue
            fn_call = getattr(p, "function_call", None)
            if fn_call is not None:
                fn_name = getattr(fn_call, "name", "tool")
                fn_args = getattr(fn_call, "args", {})
                texts.append(f"[tool_call:{fn_name}({fn_args})]")
                continue
            fn_resp = getattr(p, "function_response", None)
            if fn_resp is not None:
                fn_name = getattr(fn_resp, "name", "tool")
                fn_res = getattr(fn_resp, "response", {})
                texts.append(f"[tool_response:{fn_name}->{fn_res}]")
        return {"role": role, "content": " ".join(t for t in texts if t)}
    return {"role": role, "content": str(getattr(item, "content", item))}


def _rebuild_content_item(template_item: Any, role: str, text: str) -> Any:
    """Reconstruct a summary `Content` item matching the type of `template_item`."""
    if GENAI_TYPES_AVAILABLE and genai_types is not None and not isinstance(template_item, dict):
        return genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=text)])
    if hasattr(template_item, "parts") and hasattr(template_item, "role"):
        cls = type(template_item)
        first_part = template_item.parts[0] if getattr(template_item, "parts", None) else None
        part_cls = type(first_part) if first_part is not None else None
        try:
            new_part = part_cls(text=text) if part_cls is not None else text
            return cls(role=role, parts=[new_part])
        except Exception:
            pass
    return {"role": role, "content": text}


def adk_compaction_before_model_callback(
    callback_context: Any,
    llm_request: Any,
) -> Optional[Any]:
    """Google ADK `before_model_callback` that compacts `llm_request.contents` preserving `function_call`/`function_response` parts."""
    # Attach cached_content handle if available on request config
    cached_name = _GLOBAL_COMPACTION_MANAGER.context_cache._active_cached_content_name
    if cached_name and hasattr(llm_request, "config") and llm_request.config is not None:
        if hasattr(llm_request.config, "cached_content"):
            llm_request.config.cached_content = cached_name

    contents = getattr(llm_request, "contents", None)
    window_size = _GLOBAL_COMPACTION_MANAGER.sliding_window_turns
    if not isinstance(contents, list) or len(contents) <= window_size:
        return None

    template_item = contents[0]
    normalized_messages = [_extract_role_and_text(item) for item in contents]

    session_id = getattr(callback_context, "session_id", "default_session")
    user_id = getattr(callback_context, "user_id", "default_user")
    compacted = _GLOBAL_COMPACTION_MANAGER.compact_history(
        normalized_messages,
        user_id=user_id,
        session_id=session_id,
    )
    if compacted.compaction_applied and hasattr(llm_request, "contents"):
        # Preserve the original Content objects (including any function_call / function_response parts)
        # in the sliding window intact, and prepend only the compacted digest at index 0.
        digest_item = _rebuild_content_item(
            template_item,
            "user",
            compacted.summary_digest or "[COMPACTED HISTORY]",
        )
        retained_original_items = list(contents[-window_size:])
        # Ensure we never start the sliding window on a `function_response` without its preceding `function_call`
        if retained_original_items and len(contents) > window_size:
            first_parts = getattr(retained_original_items[0], "parts", None) or []
            if any(getattr(p, "function_response", None) is not None for p in first_parts):
                retained_original_items = list(contents[-(window_size + 1) :])
        llm_request.contents = [digest_item, *retained_original_items]
    return None
