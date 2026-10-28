from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

SAMPLE = {
    "LIMIT_BAL": 50000.0, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "repayment_status_1": 0, "repayment_status_2": 0, "repayment_status_3": 0,
    "repayment_status_4": 0, "repayment_status_5": 0, "repayment_status_6": 0,
    "utilisation_ratio_1": 0.4, "utilisation_ratio_2": 0.4, "utilisation_ratio_3": 0.4,
    "utilisation_ratio_4": 0.4, "utilisation_ratio_5": 0.4, "utilisation_ratio_6": 0.4,
    "payment_ratio_1": 0.1, "payment_ratio_2": 0.1, "payment_ratio_3": 0.1,
    "payment_ratio_4": 0.1, "payment_ratio_5": 0.1, "payment_ratio_6": 0.1,
}


def pd_of(payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    return float(r.json()["Probability_of_Default"].rstrip("%"))


def test_predict_responds():
    assert client.post("/predict", json=SAMPLE).status_code == 200


def test_response_has_expected_fields():
    body = client.post("/predict", json=SAMPLE).json()
    for key in ("Probability_of_Default", "Expected_Loss", "Decision",
                "Top_Risk_Driver", "Driver_Impact"):
        assert key in body


def test_pd_is_a_percentage():
    assert 0.0 <= pd_of(SAMPLE) <= 100.0


def test_missing_field_is_rejected():
    bad = {k: v for k, v in SAMPLE.items() if k != "AGE"}
    assert client.post("/predict", json=bad).status_code == 422


def test_worse_repayment_history_raises_pd():
    risky = dict(SAMPLE, repayment_status_1=3, repayment_status_2=3)
    assert pd_of(risky) > pd_of(SAMPLE)