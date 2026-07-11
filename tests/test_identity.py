from __future__ import annotations

import pytest

from parakh.identity import BorrowerRef, canonical_borrower_id, resolve

# The canonical id for Sharma Kirana Store's GSTIN. This constant is the shared
# contract across PARAKH and TRINETRA: both repos derive this exact id from the
# same GSTIN, which is what lets origination hand off to monitoring without a
# manual join. The identical assertion lives in the TRINETRA identity test.
SHARMA_GSTIN = "23ABCDE1234F1Z5"
SHARMA_CANONICAL_ID = "brw_99200b2d0b99"


def test_sharma_canonical_id_is_stable():
    assert canonical_borrower_id({"gstin": SHARMA_GSTIN}) == SHARMA_CANONICAL_ID


def test_gstin_normalisation_is_case_and_space_insensitive():
    base = canonical_borrower_id({"gstin": SHARMA_GSTIN})
    assert canonical_borrower_id({"gstin": "  23abcde1234f1z5 "}) == base
    assert canonical_borrower_id({"gstin": "23ABCDE1234F1Z5"}) == base


def test_id_has_prefix_and_length():
    canonical_id = canonical_borrower_id({"gstin": SHARMA_GSTIN})
    assert canonical_id.startswith("brw_")
    assert len(canonical_id) == len("brw_") + 12


def test_falls_back_to_pan_then_name_state():
    from_pan = canonical_borrower_id({"pan": "ABCDE1234F"})
    assert from_pan.startswith("brw_")
    from_name = canonical_borrower_id({"name": "Sharma Kirana Store", "state": "MP"})
    assert from_name.startswith("brw_")
    assert from_pan != from_name


def test_gstin_takes_priority_over_pan():
    with_both = canonical_borrower_id({"gstin": SHARMA_GSTIN, "pan": "ZZZZZ9999Z"})
    assert with_both == SHARMA_CANONICAL_ID


def test_empty_identity_is_rejected():
    with pytest.raises(ValueError):
        canonical_borrower_id({})


def test_resolve_returns_ref():
    ref = resolve({"gstin": SHARMA_GSTIN, "name": "Sharma Kirana Store"})
    assert isinstance(ref, BorrowerRef)
    assert ref.canonical_id == SHARMA_CANONICAL_ID
    assert ref.gstin == SHARMA_GSTIN
    assert ref.display_name == "Sharma Kirana Store"
