"""Teacher cost monitoring for DVAS.

Tracks API costs for teacher models with budget management
and cost alerting capabilities.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dvas.observability.collector import get_metrics
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CostBudget:
    """Cost budget configuration."""

    daily_usd: float = 100.0
    hourly_usd: float = 10.0
    per_request_usd: float = 1.0


class TeacherCostMonitor:
    """Monitor teacher model API costs.

    Tracks spending per model with budget thresholds and alerts.

    Usage::

        monitor = TeacherCostMonitor(CostBudget(daily_usd=500.0))
        monitor.record_cost("gpt-5.5", 0.05)
        daily = monitor.get_daily_cost()
    """

    def __init__(
        self,
        budget: Optional[CostBudget] = None,
    ) -> None:
        self.budget = budget or CostBudget()
        self._costs: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def record_cost(
        self,
        model_name: str,
        cost_usd: float,
        request_type: str = "annotation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a cost measurement.

        Args:
            model_name: Name of the teacher model
            cost_usd: Cost in USD
            request_type: Type of request (annotation, batch, etc.)
            metadata: Optional additional metadata
        """
        entry = {
            "timestamp": time.time(),
            "cost_usd": cost_usd,
            "request_type": request_type,
            "metadata": metadata or {},
        }

        with self._lock:
            if model_name not in self._costs:
                self._costs[model_name] = []
            self._costs[model_name].append(entry)

        # Record in global metrics
        get_metrics().increment(
            "teacher_cost_usd",
            value=cost_usd,
            labels={"model": model_name, "request_type": request_type},
        )
        get_metrics().increment(
            "teacher_requests_total",
            labels={"model": model_name, "type": request_type},
        )

        # Check budget thresholds
        self._check_budgets(model_name, cost_usd)

    def _check_budgets(self, model_name: str, cost_usd: float) -> None:
        """Check if costs exceed budget thresholds."""
        hourly = self.get_hourly_cost(model_name)
        if hourly > self.budget.hourly_usd:
            self._trigger_alert(
                "hourly_budget_exceeded",
                {
                    "model": model_name,
                    "hourly_cost_usd": hourly,
                    "budget_usd": self.budget.hourly_usd,
                    "severity": "warning",
                },
            )

        daily = self.get_daily_cost(model_name)
        if daily > self.budget.daily_usd:
            self._trigger_alert(
                "daily_budget_exceeded",
                {
                    "model": model_name,
                    "daily_cost_usd": daily,
                    "budget_usd": self.budget.daily_usd,
                    "severity": "critical",
                },
            )

        if cost_usd > self.budget.per_request_usd:
            self._trigger_alert(
                "per_request_budget_exceeded",
                {
                    "model": model_name,
                    "cost_usd": cost_usd,
                    "budget_usd": self.budget.per_request_usd,
                    "severity": "warning",
                },
            )

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        logger.warning(
            "teacher_cost_alert",
            alert_type=alert_type,
            **details,
        )
        for handler in self._alert_handlers:
            try:
                handler(alert_type, details)
            except Exception as e:
                logger.error("alert_handler_failed", error=str(e))

    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Remove an alert handler.

        Returns:
            True if handler was found and removed
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
            return True
        return False

    def get_hourly_cost(self, model_name: Optional[str] = None) -> float:
        """Get cost for the last hour.

        Args:
            model_name: Optional model filter (all models if None)

        Returns:
            Total cost in USD for the last hour
        """
        cutoff = time.time() - 3600
        return self._get_cost_since(cutoff, model_name)

    def get_daily_cost(self, model_name: Optional[str] = None) -> float:
        """Get cost for the last 24 hours.

        Args:
            model_name: Optional model filter (all models if None)

        Returns:
            Total cost in USD for the last 24 hours
        """
        cutoff = time.time() - 86400
        return self._get_cost_since(cutoff, model_name)

    def _get_cost_since(self, timestamp: float, model_name: Optional[str] = None) -> float:
        """Get total cost since a timestamp."""
        total = 0.0
        with self._lock:
            models = [model_name] if model_name else list(self._costs.keys())
            for model in models:
                for entry in self._costs.get(model, []):
                    if entry["timestamp"] >= timestamp:
                        total += entry["cost_usd"]
        return total

    def get_cost_by_model(self) -> Dict[str, float]:
        """Get total cost per model.

        Returns:
            Dict mapping model names to total cost in USD
        """
        result: Dict[str, float] = {}
        with self._lock:
            for model, entries in self._costs.items():
                result[model] = sum(e["cost_usd"] for e in entries)
        return result

    def get_cost_breakdown(self, model_name: str) -> Dict[str, float]:
        """Get cost breakdown by request type.

        Args:
            model_name: Name of the teacher model

        Returns:
            Dict mapping request types to total cost
        """
        breakdown: Dict[str, float] = {}
        with self._lock:
            for entry in self._costs.get(model_name, []):
                req_type = entry["request_type"]
                breakdown[req_type] = breakdown.get(req_type, 0.0) + entry["cost_usd"]
        return breakdown

    def get_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive cost statistics.

        Args:
            model_name: Optional model filter

        Returns:
            Dict with hourly, daily, total costs and budget status
        """
        hourly = self.get_hourly_cost(model_name)
        daily = self.get_daily_cost(model_name)
        by_model = self.get_cost_by_model()
        total = sum(by_model.values())

        return {
            "hourly_cost_usd": hourly,
            "daily_cost_usd": daily,
            "total_cost_usd": total,
            "hourly_budget_usd": self.budget.hourly_usd,
            "daily_budget_usd": self.budget.daily_usd,
            "hourly_utilization": hourly / max(self.budget.hourly_usd, 0.01),
            "daily_utilization": daily / max(self.budget.daily_usd, 0.01),
            "by_model": by_model,
        }

    def is_within_budget(self, model_name: Optional[str] = None) -> bool:
        """Check if costs are within daily budget.

        Args:
            model_name: Optional model filter

        Returns:
            True if daily cost is within budget
        """
        return self.get_daily_cost(model_name) <= self.budget.daily_usd

    def get_most_expensive_models(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get the n most expensive models.

        Args:
            n: Number of models to return

        Returns:
            List of dicts with model name and total cost
        """
        by_model = self.get_cost_by_model()
        sorted_models = sorted(by_model.items(), key=lambda x: x[1], reverse=True)
        return [{"model": name, "total_cost_usd": cost} for name, cost in sorted_models[:n]]

    def estimate_remaining_budget(self) -> Dict[str, Any]:
        """Estimate remaining budget for the day.

        Returns:
            Dict with remaining budget and estimated requests
        """
        daily = self.get_daily_cost()
        remaining = max(0, self.budget.daily_usd - daily)
        avg_cost = daily / max(len(self._costs), 1)
        estimated_requests = int(remaining / max(avg_cost, 0.001))

        return {
            "daily_budget_usd": self.budget.daily_usd,
            "daily_spent_usd": daily,
            "remaining_usd": remaining,
            "estimated_remaining_requests": estimated_requests,
        }

    def reset(self, model_name: Optional[str] = None) -> None:
        """Reset cost data.

        Args:
            model_name: Optional model to reset (all if None)
        """
        with self._lock:
            if model_name:
                self._costs.pop(model_name, None)
            else:
                self._costs.clear()
