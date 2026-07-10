from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from parakh.scoring.card import ALL_FEATURES
from parakh.synth.persona import BANK_FEATURES, EPFO_FEATURES, GST_FEATURES

RawFeed = dict[str, Any]
FeatureVector = dict[str, float]

RAIL_FEATURES: dict[str, list[str]] = {
    "aa": BANK_FEATURES,
    "gst": GST_FEATURES,
    "epfo": EPFO_FEATURES,
}


class AdapterError(ValueError):
    """Raised when a raw feed cannot be mapped to a valid feature vector."""


class ConsentedFeedAdapter(ABC):
    """Map a consented rail's raw payload to the model feature schema.

    Every rail (Account Aggregator, GST, EPFO) has a distinct raw schema. A
    concrete adapter owns exactly one direction: raw rail payload in, a subset of
    the model feature vector out. ``to_feature_vector`` merges the per-rail
    subsets and is the only method the scoring layer calls, so it validates that
    the merged vector is complete and finite before it reaches the model.

    Concrete adapters implement the three ``map_*`` methods. Each receives the raw
    payload for its rail and returns the feature subset that rail is responsible
    for. Splitting by rail keeps the consent boundary explicit: if a borrower
    withholds one rail, only that adapter's contribution is absent, which is what
    the source-ablation ladder measures.
    """

    #: Human-readable rail names, in the order sources stack in the ablation ladder.
    source_name: str = "consented-feed"

    @abstractmethod
    def map_gst(self, raw: RawFeed) -> FeatureVector:
        """Map a raw GST returns payload to the GST feature subset."""

    @abstractmethod
    def map_account_aggregator(self, raw: RawFeed) -> FeatureVector:
        """Map a raw Account Aggregator bank-statement payload to the bank subset."""

    @abstractmethod
    def map_epfo(self, raw: RawFeed) -> FeatureVector:
        """Map a raw EPFO establishment payload to the workforce subset."""

    def to_feature_vector(self, raw: RawFeed) -> FeatureVector:
        """Assemble and validate the full model feature vector from a raw feed.

        ``raw`` carries one key per consented rail (``gst``, ``aa``, ``epfo``).
        Each rail's sub-payload is mapped independently and the results merged.
        The merged vector must cover exactly ``ALL_FEATURES`` with finite values;
        anything else is an :class:`AdapterError` rather than a silent NaN that
        would reach the model.
        """
        vector: FeatureVector = {}
        vector.update(self.map_gst(raw.get("gst", {})))
        vector.update(self.map_account_aggregator(raw.get("aa", {})))
        vector.update(self.map_epfo(raw.get("epfo", {})))
        return validate_feature_vector(vector)


def validate_feature_vector(vector: FeatureVector) -> FeatureVector:
    """Return ``vector`` unchanged if it is a complete, finite feature vector.

    Enforces the same contract the API's ``/score`` guard enforces, one layer
    earlier: no missing features, no unknown features, no non-finite values.
    """
    provided = set(vector)
    known = set(ALL_FEATURES)
    missing = sorted(known - provided)
    unknown = sorted(provided - known)
    non_finite = sorted(
        name for name in provided & known if not math.isfinite(float(vector[name]))
    )
    problems: list[str] = []
    if missing:
        problems.append(f"missing features: {missing}")
    if unknown:
        problems.append(f"unknown features: {unknown}")
    if non_finite:
        problems.append(f"non-finite values: {non_finite}")
    if problems:
        raise AdapterError("; ".join(problems))
    return {name: float(vector[name]) for name in ALL_FEATURES}
