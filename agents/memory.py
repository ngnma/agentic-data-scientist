import json
import os
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class JSONMemory:
    """
    Lightweight persistent memory for the agent.
    Stores dataset fingerprint -> best model/metrics, notes,
    and reflection history used by the reflector.
    """

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {"datasets": {}, "notes": []}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup)
            self.data = {
                "datasets": {},
                "notes": [{"ts": now_iso(), "msg": f"Memory reset; backup at {backup}"}],
            }

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_dataset_record(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self.data.get("datasets", {}).get(fingerprint)

    def upsert_dataset_record(self, fingerprint: str, record: Dict[str, Any]) -> None:
        existing = self.data.setdefault("datasets", {}).get(fingerprint, {})
        existing.update(record)
        self.data.setdefault("datasets", {})[fingerprint] = existing
        self.save()

    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()

    def get_reflection_memory(self, fingerprint: str) -> Dict[str, Any]:
        record = self.get_dataset_record(fingerprint) or {}
        reflection_memory = record.get("reflection_memory", {})
        return {
            "history": list(reflection_memory.get("history", [])),
            "policy": dict(reflection_memory.get("policy", {})),
        }

    def build_reflection_context(
        self,
        fingerprint: str,
        runtime_flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reflection_memory = self.get_reflection_memory(fingerprint)
        context: Dict[str, Any] = {"history": reflection_memory.get("history", [])}
        context.update(reflection_memory.get("policy", {}))
        if runtime_flags:
            context.update(runtime_flags)
        return context

    def store_reflection(
        self,
        fingerprint: str,
        reflection_entry: Dict[str, Any],
        max_entries: int = 100,
    ) -> None:
        record = self.data.setdefault("datasets", {}).setdefault(fingerprint, {})
        reflection_memory = record.setdefault("reflection_memory", {"history": [], "policy": {}})
        history = reflection_memory.setdefault("history", [])
        history.append(reflection_entry)
        if len(history) > max_entries:
            del history[:-max_entries]
        self.save()


def build_reflection_entry(
    run_id: str,
    dataset_profile: Dict[str, Any],
    best_metrics: Dict[str, Any],
    reflection: Dict[str, Any],
    actions_applied: Optional[List[str]] = None,
    before_metrics: Optional[Dict[str, Any]] = None,
    after_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    before_metrics = before_metrics or best_metrics or {}
    after_metrics = after_metrics or best_metrics or {}

    before_f1 = float(before_metrics.get("f1_macro", 0.0) or 0.0)
    after_f1 = float(after_metrics.get("f1_macro", 0.0) or 0.0)
    before_bal_acc = float(before_metrics.get("balanced_accuracy", 0.0) or 0.0)
    after_bal_acc = float(after_metrics.get("balanced_accuracy", 0.0) or 0.0)

    return {
        "ts": now_iso(),
        "run_id": run_id,
        "context": {
            "best_model": best_metrics.get("model"),
            "rows": dataset_profile.get("shape", {}).get("rows"),
            "cols": dataset_profile.get("shape", {}).get("cols"),
            "imbalance_ratio": dataset_profile.get("imbalance_ratio"),
            "noise_ratio": dataset_profile.get("noise_ratio"),
            "missing_max": max(dataset_profile.get("missing_pct", {}).values(), default=0.0),
        },
        "issues": list(reflection.get("issues", [])),
        "suggestions": [list(s) for s in reflection.get("suggestions", [])],
        "actions_applied": list(actions_applied or []),
        "before_metrics": {
            "model": before_metrics.get("model"),
            "f1_macro": before_f1,
            "balanced_accuracy": before_bal_acc,
        },
        "after_metrics": {
            "model": after_metrics.get("model"),
            "f1_macro": after_f1,
            "balanced_accuracy": after_bal_acc,
        },
        "outcome": {
            "delta_f1_macro": after_f1 - before_f1,
            "delta_balanced_accuracy": after_bal_acc - before_bal_acc,
            "improved": (after_f1 - before_f1) > 0.0 or (after_bal_acc - before_bal_acc) > 0.0,
        },
    }


def get_relevant_reflections(
    reflection_memory: Optional[Dict[str, Any]],
    issues: List[str],
    dataset_profile: Dict[str, Any],
    best_model: Optional[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if not reflection_memory:
        return []

    history = reflection_memory.get("history", [])
    if not history or not issues:
        return []

    rows = dataset_profile.get("shape", {}).get("rows")
    cols = dataset_profile.get("shape", {}).get("cols")
    current_issue_set = set(issues)
    ranked: List[Any] = []

    for item in history:
        past_issues = set(item.get("issues", []))
        overlap = len(current_issue_set.intersection(past_issues))
        if overlap == 0:
            continue

        score = float(overlap)
        context = item.get("context", {})

        if best_model and context.get("best_model") == best_model:
            score += 1.0

        past_rows = context.get("rows")
        if isinstance(rows, (int, float)) and isinstance(past_rows, (int, float)):
            if max(rows, past_rows) > 0:
                ratio = min(rows, past_rows) / max(rows, past_rows)
                score += ratio

        past_cols = context.get("cols")
        if isinstance(cols, (int, float)) and isinstance(past_cols, (int, float)):
            if max(cols, past_cols) > 0:
                ratio = min(cols, past_cols) / max(cols, past_cols)
                score += 0.5 * ratio

        score += float(item.get("outcome", {}).get("delta_f1_macro", 0.0) or 0.0)
        ranked.append((score, item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:top_k]]


def prioritize_suggestions_from_memory(
    suggestions: List[List[str]],
    relevant_reflections: List[Dict[str, Any]],
) -> List[List[str]]:
    if not suggestions or not relevant_reflections:
        return suggestions

    action_scores: Dict[str, float] = {}
    for item in relevant_reflections:
        outcome = item.get("outcome", {})
        delta = float(outcome.get("delta_f1_macro", 0.0) or 0.0)
        improved = bool(outcome.get("improved", False))

        for action in item.get("actions_applied", []):
            action_scores.setdefault(action, 0.0)
            action_scores[action] += delta if improved else -0.05

    if not action_scores:
        return suggestions

    def action_score(action: str) -> float:
        return action_scores.get(action, 0.0)

    reordered: List[List[str]] = []
    for suggestion_group in suggestions:
        ordered_group = sorted(suggestion_group, key=action_score, reverse=True)
        reordered.append(ordered_group)

    reordered.sort(
        key=lambda group: max((action_score(action) for action in group), default=0.0),
        reverse=True,
    )
    return reordered