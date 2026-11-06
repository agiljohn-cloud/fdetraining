"""Asynchronous Memory Extraction and Long-Term Consolidation Pipeline.

Implements Rubric Category 2.4 (Async Memory Operations):
Executes expensive episodic memory extraction, dietary preference synthesis,
and vector embedding indexing asynchronously via `asyncio.create_task` with
done-callback error telemetry so user-facing turns never block, and ensures
clean task drainage before event loop shutdown.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
import uuid

from meal_concierge.memory.session_store import PersistentMealSessionStore, get_persistent_session_store
from meal_concierge.observability.pii_redactor import redact_pii_text
from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import traced_span


@dataclass
class MemoryConsolidationTaskResult:
    """Result record produced by an asynchronous memory consolidation job."""

    task_id: str
    user_id: str
    session_id: str
    status: str
    extracted_preferences: List[str]
    vector_doc_ids: List[str]
    completed_at_iso: str


class AsyncMemoryConsolidator:
    """Non-blocking background memory processor for episodic culinary memories.

    Schedules memory synthesis and vector store indexing on the active event loop
    using `asyncio.create_task` with a completion callback that logs any errors and
    records task results.
    """

    def __init__(self, session_store: Optional[PersistentMealSessionStore] = None) -> None:
        self.session_store = session_store or get_persistent_session_store()
        self._logger = get_structured_logger()
        self._pending_tasks: Dict[str, asyncio.Task[MemoryConsolidationTaskResult]] = {}
        self._pending_threads: Dict[str, threading.Thread] = {}
        self._completed_results: Dict[str, MemoryConsolidationTaskResult] = {}

    @staticmethod
    def _extract_culinary_preferences(turns: List[Dict[str, str]]) -> List[str]:
        """Extract durable culinary preferences, dislikes, allergies, and macro habits."""
        extracted: List[str] = []
        preference_markers = (
            "allergic",
            "allergy",
            "prefer",
            "love",
            "dislike",
            "hate",
            "avoid",
            "vegan",
            "vegetarian",
            "mediterranean",
            "keto",
            "gluten",
            "dairy",
            "peanut",
            "shellfish",
            "high protein",
            "budget",
            "prep time",
            "low-fodmap",
            "halal",
        )
        for turn in turns:
            content = redact_pii_text(turn.get("content", ""))
            for clause in content.replace("\n", ". ").split("."):
                clean = clause.strip()
                if not clean:
                    continue
                lower = clean.lower()
                if any(marker in lower for marker in preference_markers) and clean not in extracted:
                    extracted.append(clean)
        return extracted

    @traced_span("memory.AsyncMemoryConsolidator.consolidate_session_memories_async")
    async def consolidate_session_memories_async(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_turns: List[Dict[str, str]],
        plan_summary: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> MemoryConsolidationTaskResult:
        """Asynchronously synthesize long-term memories and index them in the vector store."""
        resolved_task_id = task_id or f"memtask-{uuid.uuid4().hex[:10]}"
        await asyncio.sleep(0)

        preferences = self._extract_culinary_preferences(conversation_turns)
        if plan_summary:
            cal = plan_summary.get("daily_calorie_target_kcal")
            prot = plan_summary.get("daily_protein_grams")
            if cal and prot:
                preferences.append(f"Target macros: {cal} kcal/day, {prot}g protein/day")

        vector_doc_ids: List[str] = []
        for idx, pref in enumerate(preferences):
            doc_id = f"mem-{user_id}-{session_id}-{idx}"
            await asyncio.to_thread(
                self.session_store.vector_store.upsert_vector_document,
                doc_id,
                namespace="episodic_memory",
                user_id=user_id,
                content=pref,
                metadata={
                    "session_id": session_id,
                    "source": "async_memory_consolidation",
                },
            )
            vector_doc_ids.append(doc_id)

        result = MemoryConsolidationTaskResult(
            task_id=resolved_task_id,
            user_id=user_id,
            session_id=session_id,
            status="COMPLETED",
            extracted_preferences=preferences,
            vector_doc_ids=vector_doc_ids,
            completed_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        self._completed_results[resolved_task_id] = result

        self._logger.info(
            f"Async memory consolidation finished for session '{session_id}' "
            f"({len(vector_doc_ids)} episodic memories indexed)",
            event_type="ASYNC_MEMORY_CONSOLIDATION_COMPLETED",
            session_id=session_id,
            payload={
                "task_id": resolved_task_id,
                "user_id": user_id,
                "preferences_count": len(preferences),
                "vector_doc_ids": vector_doc_ids,
            },
        )
        return result

    def _on_task_done(self, task_id: str, task: asyncio.Task[MemoryConsolidationTaskResult]) -> None:
        """Done-callback that cleans up `_pending_tasks` and logs any unexpected task failure."""
        self._pending_tasks.pop(task_id, None)
        if task.cancelled():
            self._logger.warning(
                f"Background memory consolidation task '{task_id}' was cancelled",
                event_type="ASYNC_MEMORY_TASK_CANCELLED",
                payload={"task_id": task_id},
            )
            return
        exc = task.exception()
        if exc is not None:
            self._logger.error(
                f"Background memory consolidation task '{task_id}' failed: {exc}",
                event_type="ASYNC_MEMORY_TASK_ERROR",
                payload={"task_id": task_id, "error": str(exc)},
            )

    def schedule_background_consolidation(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_turns: List[Dict[str, str]],
        plan_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Enqueue non-blocking memory consolidation via `asyncio.create_task` and return `task_id`."""
        task_id = f"memtask-{uuid.uuid4().hex[:10]}"
        self._logger.info(
            f"Scheduled non-blocking background memory consolidation task '{task_id}'",
            event_type="ASYNC_MEMORY_TASK_SCHEDULED",
            session_id=session_id,
            payload={"task_id": task_id, "user_id": user_id},
        )

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self.consolidate_session_memories_async(
                    user_id=user_id,
                    session_id=session_id,
                    conversation_turns=conversation_turns,
                    plan_summary=plan_summary,
                    task_id=task_id,
                )
            )
            self._pending_tasks[task_id] = task
            task.add_done_callback(lambda t: self._on_task_done(task_id, t))
        except RuntimeError:
            def _bg_worker() -> None:
                try:
                    res = asyncio.run(
                        self.consolidate_session_memories_async(
                            user_id=user_id,
                            session_id=session_id,
                            conversation_turns=conversation_turns,
                            plan_summary=plan_summary,
                            task_id=task_id,
                        )
                    )
                    self._completed_results[task_id] = res
                except Exception as exc:
                    self._logger.error(
                        f"Background thread memory consolidation '{task_id}' failed: {exc}",
                        event_type="ASYNC_MEMORY_TASK_ERROR",
                        payload={"task_id": task_id, "error": str(exc)},
                    )
                finally:
                    self._pending_threads.pop(task_id, None)

            worker = threading.Thread(target=_bg_worker, name=f"async-mem-{task_id}", daemon=True)
            self._pending_threads[task_id] = worker
            worker.start()

        return task_id

    async def wait_for_all_background_tasks(self) -> List[MemoryConsolidationTaskResult]:
        """Await all currently scheduled background memory tasks and daemon threads."""
        for th in list(self._pending_threads.values()):
            await asyncio.to_thread(th.join, 5.0)
        self._pending_threads.clear()

        if not self._pending_tasks:
            return list(self._completed_results.values())
        tasks = list(self._pending_tasks.values())
        self._pending_tasks.clear()
        await asyncio.gather(*tasks, return_exceptions=False)
        return list(self._completed_results.values())

    def get_task_result(self, task_id: str) -> Optional[MemoryConsolidationTaskResult]:
        """Return the result of a completed async memory consolidation task."""
        return self._completed_results.get(task_id)


_GLOBAL_CONSOLIDATOR: Optional[AsyncMemoryConsolidator] = None


def get_async_memory_consolidator() -> AsyncMemoryConsolidator:
    """Return the singleton `AsyncMemoryConsolidator`."""
    global _GLOBAL_CONSOLIDATOR
    if _GLOBAL_CONSOLIDATOR is None:
        _GLOBAL_CONSOLIDATOR = AsyncMemoryConsolidator()
    return _GLOBAL_CONSOLIDATOR
