import pytest
import json
import sys
from pathlib import Path

# Ensure project root is in path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.api_server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    assert b"online" in response.data

def test_index_redirect(client):
    """Test the API documentation/fallback dashboard endpoint."""
    response = client.get('/')
    assert response.status_code == 200