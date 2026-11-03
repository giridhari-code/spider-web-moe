"""Re-export ActiveSet from spider_moe (single source of truth)."""
from spider_moe.active_set import ActiveSet, compact_active, scatter_active

__all__ = ["ActiveSet", "compact_active", "scatter_active"]
