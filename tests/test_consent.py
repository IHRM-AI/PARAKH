import pytest

from parakh.consent.artefact import create_consent


def test_one_time_consent():
    artefact = create_consent("103")
    assert artefact.purpose_code == "103"
    assert artefact.fetch_type == "ONETIME"
    assert artefact.fi_types == ["DEPOSIT"]


def test_periodic_consent_enables_monitoring():
    artefact = create_consent("104")
    assert artefact.fetch_type == "PERIODIC"
    assert artefact.frequency_value == 1


def test_unknown_purpose_is_rejected():
    with pytest.raises(ValueError):
        create_consent("999")
