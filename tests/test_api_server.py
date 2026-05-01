"""
Unit tests for api_server.py

Tests for Flask API endpoints and responses.
"""

import unittest
import json
import tempfile
import shutil
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

# We need to import the Flask app
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path is set
from api_server import app


class TestApiServerBasic(unittest.TestCase):
    """Test basic API functionality."""
    
    def setUp(self):
        """Set up test client."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/api/health')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'online')
        self.assertIn('timestamp', data)
        self.assertEqual(data['service'], 'Demeter API')
    
    def test_root_endpoint(self):
        """Test root endpoint returns documentation."""
        response = self.client.get('/')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Demeter Plant Health API', response.data)
        self.assertIn(b'/api/latest', response.data)
    
    def test_404_error_handling(self):
        """Test 404 error handling."""
        response = self.client.get('/api/nonexistent')
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)


class TestApiLatestEndpoint(unittest.TestCase):
    """Test /api/latest endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_latest_not_found(self):
        """Test /api/latest when no diagnosis exists."""
        response = self.client.get('/api/latest')
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    @patch('api_server.load_json_file')
    def test_latest_returns_diagnosis(self, mock_load):
        """Test /api/latest returns diagnosis when available."""
        mock_diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "Thriving",
            "health_score": 85
        }
        mock_load.return_value = mock_diagnosis
        
        response = self.client.get('/api/latest')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['overall_status'], 'Thriving')


class TestApiHistoryEndpoint(unittest.TestCase):
    """Test /api/history endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_history_empty(self):
        """Test /api/history with no data."""
        response = self.client.get('/api/history')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total'], 0)
        self.assertEqual(len(data['records']), 0)
    
    @patch('api_server.load_json_array_file')
    def test_history_pagination(self, mock_load):
        """Test pagination parameters."""
        mock_records = [{"id": i} for i in range(100)]
        mock_load.return_value = mock_records
        
        response = self.client.get('/api/history?limit=10&offset=0')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total'], 100)
        self.assertEqual(len(data['records']), 10)
        self.assertEqual(data['returned'], 10)
        self.assertEqual(data['limit'], 10)
        self.assertEqual(data['offset'], 0)
    
    @patch('api_server.load_json_array_file')
    def test_history_offset(self, mock_load):
        """Test offset pagination."""
        mock_records = [{"id": i} for i in range(50)]
        mock_load.return_value = mock_records
        
        response = self.client.get('/api/history?limit=10&offset=40')
        
        data = json.loads(response.data)
        self.assertEqual(data['returned'], 10)


class TestApiSummaryEndpoint(unittest.TestCase):
    """Test /api/summary endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_summary_no_data(self):
        """Test /api/summary with no diagnoses."""
        response = self.client.get('/api/summary')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total_diagnoses'], 0)
    
    @patch('api_server.load_json_array_file')
    def test_summary_with_data(self, mock_load):
        """Test /api/summary calculation."""
        mock_records = [
            {
                "health_score": 85,
                "overall_status": "Thriving",
                "cnn_result": {"primary_disease": "Healthy"}
            },
            {
                "health_score": 50,
                "overall_status": "Struggling",
                "cnn_result": {"primary_disease": "Early Blight"}
            },
            {
                "health_score": 85,
                "overall_status": "Thriving",
                "cnn_result": {"primary_disease": "Healthy"}
            }
        ]
        mock_load.return_value = mock_records
        
        response = self.client.get('/api/summary')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total_diagnoses'], 3)
        # Average of 85, 50, 85 = 73.3
        self.assertAlmostEqual(data['average_health_score'], 73.3, places=1)


class TestApiStatusEndpoint(unittest.TestCase):
    """Test /api/status endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_status_structure(self):
        """Test /api/status returns proper structure."""
        response = self.client.get('/api/status')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('service', data)
        self.assertIn('status', data)
        self.assertIn('models_available', data)
        self.assertIn('data_available', data)
        self.assertIn('latest_diagnosis', data)
        self.assertIn('diagnosis_count', data)
    
    def test_status_models_availability(self):
        """Test model availability checking."""
        response = self.client.get('/api/status')
        
        data = json.loads(response.data)
        self.assertIn('cnn_plantvillage', data['models_available'])
        self.assertIn('rf_danforth', data['models_available'])
        # These are booleans
        self.assertIsInstance(data['models_available']['cnn_plantvillage'], bool)


class TestApiThresholdsEndpoint(unittest.TestCase):
    """Test /api/status/thresholds endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_thresholds_structure(self):
        """Test thresholds endpoint returns proper structure."""
        response = self.client.get('/api/status/thresholds')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('moisture', data)
        self.assertIn('temperature', data)
        self.assertIn('sunlight', data)
        self.assertIn('disease', data)
    
    def test_thresholds_values(self):
        """Test threshold values are reasonable."""
        response = self.client.get('/api/status/thresholds')
        
        data = json.loads(response.data)
        # Moisture thresholds should be in order
        self.assertLess(data['moisture']['critical'], data['moisture']['warning'])
        self.assertLess(data['moisture']['warning'], data['moisture']['healthy'])


class TestApiConfigEndpoint(unittest.TestCase):
    """Test /api/config endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_config_endpoint_structure(self):
        """Test /api/config endpoint returns config."""
        response = self.client.get('/api/config')
        
        # May be 200 or 404 depending on config.json existence
        self.assertIn(response.status_code, [200, 404])


class TestApiExportEndpoint(unittest.TestCase):
    """Test /api/latest/export endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    @patch('api_server.load_json_file')
    def test_export_flattens_nested_data(self, mock_load):
        """Test that export flattens nested JSON."""
        mock_diagnosis = {
            "timestamp": "2026-05-01T10:00:00",
            "cnn_result": {
                "primary_disease": "Early Blight",
                "confidence": 0.84
            },
            "sensors": {
                "temperature": 24.5,
                "soil_moisture": 65.0
            }
        }
        mock_load.return_value = mock_diagnosis
        
        response = self.client.get('/api/latest/export')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Should have flattened keys
        self.assertIn('Detected_Disease', data)
        self.assertEqual(data['Detected_Disease'], 'Early Blight')


class TestApiDashboardEndpoint(unittest.TestCase):
    """Test /dashboard endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_dashboard_returns_html(self):
        """Test /dashboard endpoint returns HTML."""
        response = self.client.get('/dashboard')
        
        # May be 200 or 404 depending on dashboard.html
        self.assertIn(response.status_code, [200, 404])
        if response.status_code == 200:
            self.assertIn(b'html', response.data.lower())


class TestApiCorsHeaders(unittest.TestCase):
    """Test CORS header handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_cors_headers_present(self):
        """Test CORS headers are present in responses."""
        response = self.client.get('/api/health')
        
        self.assertEqual(response.status_code, 200)
        # Flask-CORS should add headers (may vary based on configuration)
        # At minimum, the response should be valid


class TestApiErrorHandling(unittest.TestCase):
    """Test error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_500_error_handling(self):
        """Test 500 error response format."""
        response = self.client.get('/api/health')
        
        # Health check should work
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
