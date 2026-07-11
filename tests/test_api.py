from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from parakh.api import app as api_module
from parakh.identity import canonical_borrower_id
from parakh.monitoring.trajectory import HORIZON
from parakh.scoring.card import CardService

SHARMA_GSTIN = "23ABCDE1234F1Z5"
SHARMA_CANONICAL_ID = canonical_borrower_id({"gstin": SHARMA_GSTIN})


@pytest.fixture
def client(trained_model, monkeypatch) -> TestClient:
    service = CardService(trained_model)
    monkeypatch.setattr(api_module, "_service", service)
    return TestClient(api_module.app)


@pytest.fixture
def lifespan_client(trained_model, monkeypatch) -> TestClient:
    # Enter the app via the context manager so the lifespan seeds the audit
    # ledger, which the /audit endpoints read.
    service = CardService(trained_model)
    monkeypatch.setattr(api_module, "_service", service)
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert response.headers.get("x-request-id")


def test_consent_one_time(client):
    response = client.post("/consent", json={"purpose_code": "103"})
    assert response.status_code == 200
    assert response.json()["fetch_type"] == "ONETIME"


def test_consent_rejects_unknown_purpose(client):
    response = client.post("/consent", json={"purpose_code": "999"})
    assert response.status_code == 400


def test_score_valid_input(client, valid_features):
    response = client.post("/score", json={"features": valid_features})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"] <= 100
    assert body["grade"] in {"A", "B+", "B", "C", "D"}
    assert body["reason_codes"]


def test_score_rejects_unknown_feature(client, valid_features):
    payload = {**valid_features, "not_a_feature": 1.0}
    response = client.post("/score", json={"features": payload})
    assert response.status_code == 422
    assert "unknown" in response.json()["detail"]


def test_score_rejects_missing_feature(client, valid_features):
    payload = dict(valid_features)
    payload.pop("gst_turnover_growth")
    response = client.post("/score", json={"features": payload})
    assert response.status_code == 422
    assert "missing" in response.json()["detail"]


def test_score_rejects_nan(client, valid_features):
    # A real HTTP client can send the bare `NaN` token, which Python's JSON
    # parser accepts; send it as raw content since httpx refuses to serialise it.
    features = {**valid_features, "gst_turnover_growth": "__NAN__"}
    body = json.dumps({"features": features}).replace('"__NAN__"', "NaN")
    response = client.post(
        "/score", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "non-finite" in response.json()["detail"]


def test_whatif_returns_before_and_after(client, valid_features):
    response = client.post(
        "/whatif",
        json={"features": valid_features, "adjustments": {"gst_filing_punctuality": 12.0}},
    )
    assert response.status_code == 200
    body = response.json()
    assert "before" in body and "after" in body
    assert set(body["before"]) == {"score", "grade", "limit"}


def test_monitor_is_a_real_rescore(client):
    stressed = {
        "gst_avg_monthly_turnover": 300000.0,
        "gst_turnover_growth": -0.14,
        "gst_turnover_volatility": 0.45,
        "gst_filing_punctuality": 6.0,
        "gst_turnover_decline_3m": 0.22,
        "bank_avg_monthly_credits": 190000.0,
        "bank_balance_dip_count": 6.0,
        "bank_bounce_count": 3.0,
        "bank_cash_buffer_days": 9.0,
        "bank_credit_debit_ratio": 0.82,
        "xf_gst_bank_gap": 0.40,
        "epfo_headcount": 7.0,
        "epfo_headcount_trend": -0.10,
    }
    response = client.post("/monitor", json={"features": stressed})
    assert response.status_code == 200
    body = response.json()
    assert len(body["scores"]) == HORIZON
    assert len(body["months"]) == HORIZON
    assert body["scores"][0] > body["scores"][-1]
    assert body["deteriorating"] is True


def test_memo_template_fallback(client, valid_features):
    response = client.post("/memo", json={"borrower": "Acme Traders", "features": valid_features})
    assert response.status_code == 200
    body = response.json()
    assert body["borrower"] == "Acme Traders"
    assert body["status"] == "Awaiting officer approval"
    assert "template" in body["generated_by"]
    assert body["body"]


def test_score_requires_service_when_unloaded(monkeypatch):
    monkeypatch.setattr(api_module, "_service", None)
    client = TestClient(api_module.app)
    response = client.post("/score", json={"features": {}})
    assert response.status_code == 503


def test_api_key_gate_default_open(client, valid_features):
    response = client.post("/score", json={"features": valid_features})
    assert response.status_code == 200


def test_api_key_gate_rejects_when_set(client, valid_features, monkeypatch):
    monkeypatch.setattr(api_module.settings, "api_key", "s3cret")
    unauth = client.post("/score", json={"features": valid_features})
    assert unauth.status_code == 401
    authed = client.post(
        "/score", json={"features": valid_features}, headers={"x-api-key": "s3cret"}
    )
    assert authed.status_code == 200


def _upload(name: str) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (name, b"dummy bytes", "application/pdf")}


class _FakeOcrAvailable:
    def __init__(self, *_args, **_kwargs) -> None:
        self.available = True

    def extract(self, _path) -> str:
        return (
            "Business name: Test Traders\nLocation: Pune, MH\n"
            "GSTIN 09ABCDE1234F1Z5\nGST turnover per month: 500000\n"
        )


class _FakeOcrBroken:
    def __init__(self, *_args, **_kwargs) -> None:
        self.available = True

    def extract(self, _path) -> str:
        raise RuntimeError("OCR service unreachable")


def test_extract_demo_fixture_when_ocr_offline(client, monkeypatch):
    monkeypatch.setattr(api_module.settings, "ocr_service_url", "")
    response = client.post("/extract", files=_upload("gst-sample.pdf"))
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "demo fixture — OCR offline"
    assert body["fields"]["name"] == "Gupta Provisions"


def test_extract_demo_via_query_flag(client, monkeypatch):
    monkeypatch.setattr(api_module.settings, "ocr_service_url", "")
    response = client.post("/extract?demo=true", files=_upload("anything.pdf"))
    assert response.status_code == 200
    assert response.json()["source"] == "demo fixture — OCR offline"


def test_extract_reports_offline_for_unknown_file(client, monkeypatch):
    monkeypatch.setattr(api_module.settings, "ocr_service_url", "")
    response = client.post("/extract", files=_upload("statement.pdf"))
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "unavailable"
    assert body["fields"] == {}
    assert "offline" in body["message"]


def test_extract_parses_ocr_text(client, monkeypatch):
    monkeypatch.setattr(api_module.settings, "ocr_service_url", "http://ocr.internal")
    monkeypatch.setattr(api_module, "OcrClient", _FakeOcrAvailable)
    response = client.post("/extract", files=_upload("real-return.pdf"))
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "ocr"
    assert body["fields"]["name"] == "Test Traders"
    assert body["fields"]["gstin"] == "09ABCDE1234F1Z5"
    assert body["fields"]["gst_avg_monthly_turnover"] == 500000


def test_extract_degrades_when_ocr_call_fails(client, monkeypatch):
    monkeypatch.setattr(api_module.settings, "ocr_service_url", "http://ocr.internal")
    monkeypatch.setattr(api_module, "OcrClient", _FakeOcrBroken)
    response = client.post("/extract", files=_upload("real-return.pdf"))
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "unavailable"
    assert "could not be reached" in body["message"]


def test_extract_sample_short_circuits_before_ocr(client, monkeypatch):
    # A recognised sample serves the fixture without touching OCR, so it stays
    # instant even when OCR_SERVICE_URL points at an unreachable GPU.
    monkeypatch.setattr(api_module.settings, "ocr_service_url", "http://ocr.internal")
    monkeypatch.setattr(api_module, "OcrClient", _FakeOcrBroken)
    response = client.post("/extract", files=_upload("gupta-sample.pdf"))
    assert response.status_code == 200
    assert response.json()["source"] == "demo fixture — OCR offline"


def test_resolve_returns_canonical_id_and_presence(client):
    response = client.get("/resolve", params={"gstin": SHARMA_GSTIN, "name": "Sharma Kirana Store"})
    assert response.status_code == 200
    body = response.json()
    assert body["canonical_id"] == SHARMA_CANONICAL_ID
    assert body["gstin"] == SHARMA_GSTIN
    assert body["display_name"] == "Sharma Kirana Store"
    assert body["originated"] is True
    assert body["monitored"] is True


def test_resolve_normalises_gstin(client):
    response = client.get("/resolve", params={"gstin": " 23abcde1234f1z5 "})
    assert response.status_code == 200
    assert response.json()["canonical_id"] == SHARMA_CANONICAL_ID


def test_resolve_unknown_borrower_is_not_monitored(client):
    response = client.get("/resolve", params={"gstin": "29ZZZZZ9999Z9Z9"})
    assert response.status_code == 200
    body = response.json()
    assert body["originated"] is False
    assert body["monitored"] is False


def test_audit_chain_is_ordered_and_verifies(lifespan_client):
    chain = lifespan_client.get(f"/audit/{SHARMA_CANONICAL_ID}")
    assert chain.status_code == 200
    entries = chain.json()["entries"]
    assert [entry["event_type"] for entry in entries] == [
        "CONSENT_GRANTED",
        "DATA_FETCHED",
        "SCORED",
        "DECISION",
        "MONITORING_WATCH",
    ]
    verified = lifespan_client.get(f"/audit/{SHARMA_CANONICAL_ID}/verify")
    assert verified.status_code == 200
    body = verified.json()
    assert body["ok"] is True
    assert body["first_broken_seq"] is None


def test_audit_unknown_borrower_returns_404(lifespan_client):
    assert lifespan_client.get("/audit/brw_unknown").status_code == 404
    assert lifespan_client.get("/audit/brw_unknown/verify").status_code == 404


def test_audit_returns_503_when_ledger_uninitialised(client, monkeypatch):
    monkeypatch.setattr(api_module, "_ledger", None)
    assert client.get(f"/audit/{SHARMA_CANONICAL_ID}").status_code == 503
    assert client.get(f"/audit/{SHARMA_CANONICAL_ID}/verify").status_code == 503
