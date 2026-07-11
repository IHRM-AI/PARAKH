from __future__ import annotations

from parakh.api.extraction import (
    DEMO_FIXTURE,
    demo_extraction,
    is_demo_request,
    parse_document_text,
)

SAMPLE_TEXT = """
Business name: Gupta Provisions
Location: Kanpur, UP
GSTIN 09ABCDE1234F1Z5

GST turnover per month: Rs 6,20,000
Turnover growth: 0.09
Turnover volatility 0.16
GST filings on time: 11
3-month decline: 0
Bank credits per month: 5,74,000
Month-end balance dips: 2
Cheque bounces: 0
Cash buffer days: 26
Credit/debit ratio: 1.08
GST-vs-bank gap: 0.07
EPFO headcount: 7
Headcount trend: 0.04
"""


def test_parse_extracts_gstin():
    fields = parse_document_text("Firm registered under GSTIN 27FGHIJ5678K2Z9 today.")
    assert fields["gstin"] == "27FGHIJ5678K2Z9"


def test_parse_extracts_name_and_location():
    fields = parse_document_text("Business name: Sharma Kirana Store\nLocation: Indore, MP")
    assert fields["name"] == "Sharma Kirana Store"
    assert fields["location"] == "Indore, MP"


def test_parse_extracts_labelled_numbers():
    fields = parse_document_text(SAMPLE_TEXT)
    assert fields["gst_avg_monthly_turnover"] == 620000
    assert fields["bank_avg_monthly_credits"] == 574000
    assert fields["gst_turnover_growth"] == 0.09
    assert fields["gst_filing_punctuality"] == 11
    assert fields["epfo_headcount"] == 7
    assert fields["bank_credit_debit_ratio"] == 1.08
    assert fields["xf_gst_bank_gap"] == 0.07


def test_parse_returns_only_found_fields():
    fields = parse_document_text("GSTIN 09ABCDE1234F1Z5 and nothing else useful here.")
    assert set(fields) == {"gstin"}


def test_parse_empty_text_returns_empty():
    assert parse_document_text("") == {}


def test_parse_ignores_unparseable_label_without_number():
    fields = parse_document_text("Cash buffer days: many")
    assert "bank_cash_buffer_days" not in fields


def test_is_demo_request_via_flag():
    assert is_demo_request(None, True) is True


def test_is_demo_request_via_sample_filename():
    assert is_demo_request("sample-gst-return.pdf", False) is True
    assert is_demo_request("Gupta-provisions.png", False) is True


def test_is_demo_request_rejects_unknown_file():
    assert is_demo_request("random-statement.pdf", False) is False
    assert is_demo_request(None, False) is False


def test_demo_extraction_matches_fixture():
    result = demo_extraction()
    assert result.fields == DEMO_FIXTURE
    assert "offline" in result.source
    result.fields["name"] = "mutated"
    assert DEMO_FIXTURE["name"] == "Gupta Provisions"
