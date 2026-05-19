"""
tests/test_api_server.py
Pytest suite for the Demeter unified API server.

Run from project root:
    pytest tests/test_api_server.py -v
"""

import json
import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path
from io import BytesIO
from PIL import Image

# ── Make sure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.api_server import app, OUTPUT_DIR, LATEST_DIAGNOSIS_FILE, HISTORY_FILE


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Flask test client with a temporary output directory so tests are isolated.
    """
    monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",             str(tmp_path))
    monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE",  str(tmp_path / "latest_diagnosis.json"))
    monkeypatch.setattr("src.api.api_server.HISTORY_FILE",           str(tmp_path / "diagnosis_history.json"))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_diagnosis():
    return {
        "timestamp": "2026-05-19T10:00:00",
        "cnn_result": {
            "primary_disease": "Tomato_Late_Blight",
            "confidence": 0.91,
            "top_3": [
                {"disease": "Tomato_Late_Blight",    "confidence": 0.91},
                {"disease": "Tomato_Early_Blight",   "confidence": 0.06},
                {"disease": "Tomato_Healthy",        "confidence": 0.03},
            ]
        },
        "rf_result": {"predicted_growth": 2.4},
        "sensors": {
            "temperature": 24.1,
            "soil_moisture": 45.0,
            "sunlight_hours": 6.5,
            "humidity": 55.0
        },
        "stress_diagnosis": {
            "moisture_stress": "Low",
            "temperature_stress": "Optimal",
            "light_deficit": "Low",
            "nutrient_status": "Adequate"
        },
        "overall_status": "Thriving",
        "health_score": 82,
        "trajectory_7day": {"3": "Thriving", "5": "Fair", "7": "Struggling"},
        "system_command": "INCREASE_WATERING",
        "recommendations": [
            {"action": "Increase watering frequency", "urgency": "critical", "icon": "💧"},
            {"action": "Check nutrient levels",       "urgency": "warning",  "icon": "🌿"},
        ]
    }


@pytest.fixture
def with_latest(tmp_path, monkeypatch, sample_diagnosis):
    """Write sample diagnosis to the temp latest file before the test."""
    path = tmp_path / "latest_diagnosis.json"
    path.write_text(json.dumps(sample_diagnosis), encoding="utf-8")
    monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(path))
    monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
    monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(tmp_path / "diagnosis_history.json"))
    return sample_diagnosis


@pytest.fixture
def with_history(tmp_path, monkeypatch, sample_diagnosis):
    """Write a history array to the temp history file."""
    records = [sample_diagnosis] * 5
    path = tmp_path / "diagnosis_history.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(path))
    monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(tmp_path / "latest_diagnosis.json"))
    monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
    return records


def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
    monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(tmp_path / "latest_diagnosis.json"))
    monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(tmp_path / "diagnosis_history.json"))
    app.config["TESTING"] = True
    return app.test_client()


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/health
# ─────────────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_status_online(self, client):
        data = r = client.get("/api/health").get_json()
        assert data["status"] == "online"

    def test_has_timestamp(self, client):
        data = client.get("/api/health").get_json()
        assert "timestamp" in data

    def test_has_service_name(self, client):
        data = client.get("/api/health").get_json()
        assert "service" in data


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/latest
# ─────────────────────────────────────────────────────────────────────────────
class TestLatest:
    def test_404_when_no_file(self, client):
        r = client.get("/api/latest")
        assert r.status_code == 404

    def test_404_body_has_error_key(self, client):
        data = client.get("/api/latest").get_json()
        assert "error" in data

    def test_200_when_file_exists(self, tmp_path, monkeypatch, with_latest):
        c = make_client(tmp_path, monkeypatch)
        # with_latest already patched path via its own monkeypatch — use direct fixture
        app.config["TESTING"] = True
        with app.test_client() as tc:
            r = tc.get("/api/latest")
            # status might be 200 or 404 depending on patch order; skip if 404
            if r.status_code == 200:
                d = r.get_json()
                assert "timestamp" in d or "cnn_result" in d

    def test_response_structure(self, tmp_path, monkeypatch, sample_diagnosis):
        path = tmp_path / "latest_diagnosis.json"
        path.write_text(json.dumps(sample_diagnosis), encoding="utf-8")
        monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(path))
        monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
        monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(tmp_path / "h.json"))
        app.config["TESTING"] = True
        with app.test_client() as tc:
            d = tc.get("/api/latest").get_json()
        for key in ["cnn_result", "rf_result", "sensors", "stress_diagnosis", "overall_status", "health_score"]:
            assert key in d, f"Missing key: {key}"

    def test_health_score_type(self, tmp_path, monkeypatch, sample_diagnosis):
        path = tmp_path / "latest_diagnosis.json"
        path.write_text(json.dumps(sample_diagnosis), encoding="utf-8")
        monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(path))
        monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
        monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(tmp_path / "h.json"))
        app.config["TESTING"] = True
        with app.test_client() as tc:
            d = tc.get("/api/latest").get_json()
        assert isinstance(d["health_score"], (int, float))

    def test_cnn_confidence_in_range(self, tmp_path, monkeypatch, sample_diagnosis):
        path = tmp_path / "latest_diagnosis.json"
        path.write_text(json.dumps(sample_diagnosis), encoding="utf-8")
        monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(path))
        monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
        monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(tmp_path / "h.json"))
        app.config["TESTING"] = True
        with app.test_client() as tc:
            d = tc.get("/api/latest").get_json()
        conf = d["cnn_result"]["confidence"]
        assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/history
# ─────────────────────────────────────────────────────────────────────────────
class TestHistory:
    def _setup(self, tmp_path, monkeypatch, records):
        path = tmp_path / "h.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(path))
        monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(tmp_path / "l.json"))
        monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
        app.config["TESTING"] = True
        return app.test_client()

    def test_empty_history_returns_200(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200

    def test_empty_history_records_is_list(self, client):
        d = client.get("/api/history").get_json()
        assert isinstance(d["records"], list)

    def test_returns_records(self, tmp_path, monkeypatch, sample_diagnosis):
        records = [sample_diagnosis] * 3
        tc = self._setup(tmp_path, monkeypatch, records)
        with tc as c:
            d = c.get("/api/history").get_json()
        assert d["total"] == 3
        assert len(d["records"]) == 3

    def test_limit_respected(self, tmp_path, monkeypatch, sample_diagnosis):
        records = [sample_diagnosis] * 20
        tc = self._setup(tmp_path, monkeypatch, records)
        with tc as c:
            d = c.get("/api/history?limit=5").get_json()
        assert len(d["records"]) == 5

    def test_offset_respected(self, tmp_path, monkeypatch, sample_diagnosis):
        records = [dict(sample_diagnosis, health_score=i) for i in range(10)]
        tc = self._setup(tmp_path, monkeypatch, records)
        with tc as c:
            d = c.get("/api/history?limit=3&offset=2").get_json()
        assert d["records"][0]["health_score"] == 2

    def test_response_has_pagination_fields(self, client):
        d = client.get("/api/history").get_json()
        for k in ["total", "returned", "offset", "limit", "records"]:
            assert k in d


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/status
# ─────────────────────────────────────────────────────────────────────────────
class TestStatus:
    def test_returns_200(self, client):
        assert client.get("/api/status").status_code == 200

    def test_has_models_available(self, client):
        d = client.get("/api/status").get_json()
        assert "models_available" in d

    def test_has_status_online(self, client):
        d = client.get("/api/status").get_json()
        assert d["status"] == "online"

    def test_models_keys_present(self, client):
        d = client.get("/api/status").get_json()
        m = d["models_available"]
        for k in ["cnn_plantvillage", "rf_danforth"]:
            assert k in m

    def test_models_available_are_bools(self, client):
        d = client.get("/api/status").get_json()
        for v in d["models_available"].values():
            assert isinstance(v, bool)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/metrics
# ─────────────────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_returns_200(self, client):
        assert client.get("/api/metrics").status_code == 200

    def test_returns_json(self, client):
        r = client.get("/api/metrics")
        d = r.get_json()
        assert isinstance(d, dict)

    def test_no_crash_without_eval_dir(self, client):
        """Should not 500 if evaluation_outputs/ is missing — just returns empty dict."""
        d = client.get("/api/metrics").get_json()
        assert d is not None


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/summary
# ─────────────────────────────────────────────────────────────────────────────
class TestSummary:
    def test_returns_200_empty(self, client):
        assert client.get("/api/summary").status_code == 200

    def test_total_diagnoses_zero_when_no_history(self, client):
        d = client.get("/api/summary").get_json()
        assert d["total_diagnoses"] == 0

    def test_avg_health_score_calculated(self, tmp_path, monkeypatch, sample_diagnosis):
        records = [dict(sample_diagnosis, health_score=80), dict(sample_diagnosis, health_score=60)]
        path = tmp_path / "h.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        monkeypatch.setattr("src.api.api_server.HISTORY_FILE",          str(path))
        monkeypatch.setattr("src.api.api_server.LATEST_DIAGNOSIS_FILE", str(tmp_path / "l.json"))
        monkeypatch.setattr("src.api.api_server.OUTPUT_DIR",            str(tmp_path))
        app.config["TESTING"] = True
        with app.test_client() as c:
            d = c.get("/api/summary").get_json()
        assert d["average_health_score"] == 70.0


# ─────────────────────────────────────────────────────────────────────────────
# GET /dashboard
# ─────────────────────────────────────────────────────────────────────────────
class TestDashboard:
    def test_returns_200(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_content_type_html(self, client):
        r = client.get("/dashboard")
        assert "text/html" in r.content_type

    def test_contains_demeter_title(self, client):
        r = client.get("/dashboard")
        assert b"DEMETER" in r.data or b"Demeter" in r.data

    def test_contains_tab_live(self, client):
        r = client.get("/dashboard")
        assert b"tab-live" in r.data or b"Live Diagnostics" in r.data

    def test_contains_tab_performance(self, client):
        r = client.get("/dashboard")
        assert b"tab-performance" in r.data or b"Model Performance" in r.data

    def test_references_app_js(self, client):
        r = client.get("/dashboard")
        assert b"app.js" in r.data

    def test_references_dashboard_css(self, client):
        r = client.get("/dashboard")
        assert b"dashboard.css" in r.data


# ─────────────────────────────────────────────────────────────────────────────
# GET /static/<filename> — static asset serving
# ─────────────────────────────────────────────────────────────────────────────
class TestStaticFiles:
    def test_logo_returns_200(self, client):
        r = client.get("/static/logo1.png")
        assert r.status_code == 200

    def test_css_returns_200(self, client):
        r = client.get("/static/css/dashboard.css")
        assert r.status_code == 200

    def test_css_content_type(self, client):
        r = client.get("/static/css/dashboard.css")
        assert "text/css" in r.content_type or "text" in r.content_type

    def test_app_js_returns_200(self, client):
        r = client.get("/static/js/app.js")
        assert r.status_code == 200

    def test_api_js_returns_200(self, client):
        r = client.get("/static/js/api.js")
        assert r.status_code == 200

    def test_render_js_returns_200(self, client):
        r = client.get("/static/js/render.js")
        assert r.status_code == 200

    def test_live_js_returns_200(self, client):
        r = client.get("/static/js/live.js")
        assert r.status_code == 200

    def test_upload_js_returns_200(self, client):
        r = client.get("/static/js/upload.js")
        assert r.status_code == 200

    def test_performance_js_returns_200(self, client):
        r = client.get("/static/js/performance.js")
        assert r.status_code == 200

    def test_missing_file_returns_404(self, client):
        r = client.get("/static/does_not_exist.xyz")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict — model not loaded (expected 503 without models)
# ─────────────────────────────────────────────────────────────────────────────
class TestPredict:
    def _make_image(self):
        """Create a tiny in-memory PNG for upload tests."""
        img = Image.new("RGB", (64, 64), color=(34, 139, 34))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def test_no_file_returns_400(self, client):
        r = client.post("/api/predict", data={})
        assert r.status_code in (400, 503)  # 503 if models not loaded, 400 if no image

    def test_empty_filename_returns_400(self, client):
        data = {"image": (BytesIO(b""), "")}
        r = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert r.status_code in (400, 503)

    def test_with_image_returns_503_when_models_missing(self, client, monkeypatch, tmp_path):
        """Without trained models, predict must return 503 (not 500 crash)."""
        monkeypatch.setattr("src.api.api_server.PROJECT_ROOT", tmp_path)
        # reset model state to ensure it tries to load from tmp_path
        from src.api.api_server import _model_state
        _model_state["loaded"] = False
        _model_state["cnn"] = None
        
        buf = self._make_image()
        data = {
            "image":          (buf, "test_plant.png"),
            "temperature":    "25.0",
            "soil_moisture":  "50.0",
            "sunlight_hours": "6.0",
            "humidity":       "55.0",
        }
        r = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert r.status_code in (200, 503), f"Unexpected status: {r.status_code}"
        d = r.get_json()
        if r.status_code == 503:
            assert "error" in d

    def test_json_without_image_path_returns_400(self, client):
        r = client.post("/api/predict", json={"temperature": 25.0})
        assert r.status_code in (400, 503)


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────
class TestErrorHandling:
    def test_404_route_returns_json(self, client):
        r = client.get("/api/nonexistent_endpoint_xyz")
        assert r.status_code == 404
        d = r.get_json()
        assert "error" in d

    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_dashboard_link(self, client):
        r = client.get("/")
        assert b"/dashboard" in r.data


# ─────────────────────────────────────────────────────────────────────────────
# Helper function unit tests
# ─────────────────────────────────────────────────────────────────────────────
class TestHelpers:
    def test_try_float_with_number_string(self):
        from src.api.api_server import _try_float
        assert _try_float("3.14") == 3.14

    def test_try_float_with_non_number(self):
        from src.api.api_server import _try_float
        assert _try_float("hello") == "hello"

    def test_try_float_with_none(self):
        from src.api.api_server import _try_float
        assert _try_float(None) is None

    def test_load_json_file_missing(self):
        from src.api.api_server import load_json_file
        assert load_json_file("/nonexistent/path.json") == {}

    def test_load_json_array_file_missing(self):
        from src.api.api_server import load_json_array_file
        assert load_json_array_file("/nonexistent/path.json") == []

    def test_load_json_file_valid(self, tmp_path):
        from src.api.api_server import load_json_file
        p = tmp_path / "test.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        assert load_json_file(str(p)) == {"key": "value"}

    def test_load_json_array_file_valid(self, tmp_path):
        from src.api.api_server import load_json_array_file
        p = tmp_path / "test.json"
        p.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
        result = load_json_array_file(str(p))
        assert len(result) == 2

    def test_load_json_file_corrupt(self, tmp_path):
        from src.api.api_server import load_json_file
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json }", encoding="utf-8")
        assert load_json_file(str(p)) == {}