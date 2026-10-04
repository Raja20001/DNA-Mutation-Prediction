"""
Test Suite for DatasetManager, MultiEngineMutationDetector, and Flask Endpoints.
Verifies Upload & Live Real-Time Datasets, In-Memory Training/Testing,
and Multi-Model Backend Execution with Quantum Best Model output.
"""
import io
import json
import pytest
import pandas as pd

from flask_app.app import app
from src.preprocessing.dataset_manager import DatasetManager
from src.mutation.multi_engine_detector import MultiEngineMutationDetector


class TestDatasetManager:
    def test_curated_datasets_loading(self):
        mgr = DatasetManager()
        df_cancer = mgr.get_curated_dataset("clinvar_pancancer")
        assert len(df_cancer) > 0
        assert "sequence" in df_cancer.columns
        assert "label" in df_cancer.columns
        assert "gene" in df_cancer.columns

        df_hered = mgr.get_curated_dataset("clinvar_hereditary")
        assert len(df_hered) > 0
        assert "HBB" in df_hered["gene"].values or "CFTR" in df_hered["gene"].values

        df_viral = mgr.get_curated_dataset("viral_sars_cov_2")
        assert len(df_viral) > 0
        assert any("SPIKE" in str(x).upper() for x in df_viral["variant_id"].values)

    def test_parse_fasta_upload(self):
        mgr = DatasetManager()
        fasta_text = (
            ">VAR_001_TP53_mutant\n"
            "ATGCGATCGATCGATCGATCGATCGATCGATC\n"
            ">VAR_002_TP53_wildtype\n"
            "ATGCGATCGATCGATCGATCGATCGATCGATC\n"
        )
        df = mgr.parse_uploaded_file("test.fasta", fasta_text.encode("utf-8"))
        assert len(df) == 2
        assert "sequence" in df.columns
        assert "label" in df.columns
        assert df.iloc[0]["label"] == 1
        assert df.iloc[1]["label"] == 0

    def test_parse_fastq_upload(self):
        mgr = DatasetManager()
        fastq_text = (
            "@READ_1\n"
            "ATGCGATCGATCGATC\n"
            "+\n"
            "IIIIIIIIIIIIIIII\n"
            "@READ_2\n"
            "GCTAGCTAGCTAGCTA\n"
            "+\n"
            "IIIIIIIIIIIIIIII\n"
        )
        df = mgr.parse_uploaded_file("test.fastq", fastq_text.encode("utf-8"))
        assert len(df) == 2
        assert "sequence" in df.columns

    def test_train_and_test_models(self):
        mgr = DatasetManager()
        df = mgr.get_curated_dataset("clinvar_pancancer")
        res = mgr.train_and_test_models(df, test_size=0.25)
        assert res["status"] == "success"
        assert "best_model" in res
        assert res["best_model"]["is_quantum"] is True
        assert len(res["all_models"]) >= 4
        assert "confusion_matrix" in res


class TestMultiEngineDetector:
    def test_multi_engine_execution(self):
        detector = MultiEngineMutationDetector()
        seq = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
        res = detector.run_all_models_and_select_best(seq, threshold=0.5)

        assert res["status"] == "success"
        assert res["total_models_evaluated"] == 9
        assert "best_model" in res
        assert res["best_model"]["is_quantum"] is True
        assert res["best_model"]["rank"] == 1
        assert len(res["all_models_run"]) == 9
        assert "quantum_metadata" in res["best_model"]
        assert "qubits" in res["best_model"]["quantum_metadata"]


class TestFlaskEndpoints:
    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_api_detect_multi_model(self, client):
        payload = {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
            "threshold": 0.50
        }
        resp = client.post("/api/detect", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "best_model" in data
        assert data["best_model"]["is_quantum"] is True
        assert "all_models_run" in data
        assert len(data["all_models_run"]) == 9

    def test_api_dataset_curated(self, client):
        resp = client.post("/api/dataset/curated/clinvar_pancancer")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["total_records"] > 0

    def test_api_dataset_active(self, client):
        resp = client.get("/api/dataset/active")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"

    def test_api_dataset_train_test(self, client):
        resp = client.post("/api/dataset/train_test", json={"test_size": 0.25})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "best_model" in data
        assert data["best_model"]["is_quantum"] is True

    def test_all_web_routes_ok(self, client):
        for route in ["/", "/workflow", "/concepts", "/data-studio", "/models", "/inference", "/visualizations", "/report", "/slides"]:
            resp = client.get(route)
            assert resp.status_code == 200, f"Route {route} failed with {resp.status_code}"
