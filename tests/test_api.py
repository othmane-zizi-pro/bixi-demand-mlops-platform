from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["station_count"] > 0


def test_predict_endpoint_with_local_model():
    response = client.post(
        "/predict",
        json={
            "station": "10e avenue / Masson",
            "date": "2026-01-01",
            "hour": 8,
            "is_holiday": 0,
            "temperature": 22.5,
            "feels_like": 23.0,
            "wind_speed": 12.0,
            "bad_weather": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["station"] == "10e avenue / Masson"
    assert payload["predicted_total_demand"] >= 0
    assert payload["model_source"] == "local"
