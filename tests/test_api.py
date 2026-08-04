import pytest

pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_features_endpoint_returns_list(client):
    response = client.get("/features")

    assert response.status_code == 200
    assert "expected_genes" in response.json()


def test_predict_with_full_gene_expression(client):
    features_response = client.get("/features")
    genes = features_response.json()["expected_genes"]

    fake_expression = {
        gene: 8.0
        for gene in genes
    }

    response = client.post(
        "/predict",
        json={
            "gene_expression": fake_expression
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_ad"] <= 1.0


def test_predict_handles_missing_genes_gracefully(client):
    response = client.post(
        "/predict",
        json={
            "gene_expression": {
                "APOE": 9.5
            }
        },
    )

    assert response.status_code == 200