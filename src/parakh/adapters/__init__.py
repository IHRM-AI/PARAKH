"""Integration adapters that map consented raw feeds to the model feature schema.

PARAKH scores a fixed feature vector (``parakh.scoring.card.ALL_FEATURES``). In
production those features are derived from three consented rails — Account
Aggregator bank statements, the GST returns network, and EPFO — each of which
exposes its own raw schema. An adapter is the single seam between a rail's raw
payload and the model input, so swapping the synthetic engine for a live rail is
a configuration change rather than a model change.
"""

from __future__ import annotations

from parakh.adapters.base import (
    AdapterError,
    ConsentedFeedAdapter,
    FeatureVector,
    RawFeed,
)
from parakh.adapters.persona import PersonaFeedAdapter

__all__ = [
    "AdapterError",
    "ConsentedFeedAdapter",
    "FeatureVector",
    "PersonaFeedAdapter",
    "RawFeed",
]
