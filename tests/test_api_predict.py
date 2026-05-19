import requests
from io import BytesIO
from PIL import Image

img = Image.new('RGB', (64, 64), color=(34, 139, 34))
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

from tests.test_api_server import make_client
import sys
from pathlib import Path
import json

app_dir = Path('.').resolve()
sys.path.insert(0, str(app_dir))

from src.api.api_server import app
app.config['TESTING'] = True
with app.test_client() as c:
    data = {
        'image': (buf, 'test.png'),
        'temperature': '25',
        'soil_moisture': '50',
        'sunlight_hours': '6',
        'humidity': '50'
    }
    r = c.post('/api/predict', data=data, content_type='multipart/form-data')
    print('STATUS:', r.status_code)
    print('DATA:', r.get_data(as_text=True))
