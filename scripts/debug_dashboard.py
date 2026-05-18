"""Diagnostic script — runs the /dashboard route via Flask test client and prints full traceback."""
import sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force Flask to propagate exceptions so we see the real error
import os
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = '1'

from src.api.api_server import app, PROJECT_ROOT

app.config['TESTING'] = True
app.config['PROPAGATE_EXCEPTIONS'] = False  # Let error handler run but print it

# Monkey-patch the error handler to also print traceback
import flask
original_500 = None

@app.errorhandler(500)
def debug_500(e):
    traceback.print_exc()
    return flask.jsonify({"error": "Internal server error", "message": str(e)}), 500

print("PROJECT_ROOT:", PROJECT_ROOT)
dashboard_path = PROJECT_ROOT / "src" / "frontend" / "dashboard.html"
print("Dashboard path:", dashboard_path)
print("Dashboard exists:", dashboard_path.exists())

with app.test_client() as client:
    try:
        print("\n--- Attempting GET /dashboard ---")
        response = client.get('/dashboard')
        print("Status:", response.status_code)
        if response.status_code != 200:
            print("Response body:", response.data.decode('utf-8', errors='replace'))
        else:
            print("SUCCESS - Dashboard served correctly, first 200 chars:")
            print(response.data[:200].decode('utf-8'))
    except Exception as ex:
        print("EXCEPTION during test client request:")
        traceback.print_exc()
