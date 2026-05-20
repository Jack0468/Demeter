import os
import json
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())

# 1. Fix fft_exploration.ipynb
notebook_path = PROJECT_ROOT / "notebooks" / "fft_exploration.ipynb"
if notebook_path.exists():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = f.read()
    nb_data = nb_data.replace('data/layer2_health_rgb/PlantVillage', 'data/raw/vision/PlantVillage')
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(nb_data)

# 2. Fix README.md
readme_path = PROJECT_ROOT / "README.md"
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_data = f.read()
    readme_data = readme_data.replace('data/layer2_health_rgb/PlantVillage', 'data/raw/vision/PlantVillage')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_data)

# 3. Fix SETUP_DASHBOARD.md
setup_path = PROJECT_ROOT / "SETUP_DASHBOARD.md"
if setup_path.exists():
    with open(setup_path, 'r', encoding='utf-8') as f:
        setup_data = f.read()
    setup_data = setup_data.replace('data/layer2_health_rgb/PlantVillage', 'data/raw/vision/PlantVillage')
    with open(setup_path, 'w', encoding='utf-8') as f:
        f.write(setup_data)

# 4. Fix TODO.md
todo_path = PROJECT_ROOT / "TODO.md"
if todo_path.exists():
    with open(todo_path, 'r', encoding='utf-8') as f:
        todo_data = f.read()
    todo_data = todo_data.replace('layer2_health_rgb/PlantVillage', 'raw/vision/PlantVillage')
    with open(todo_path, 'w', encoding='utf-8') as f:
        f.write(todo_data)

# 5. Fix api_server.py
api_path = PROJECT_ROOT / "src" / "api" / "api_server.py"
if api_path.exists():
    with open(api_path, 'r', encoding='utf-8') as f:
        api_data = f.read()
    api_data = api_data.replace('data/layer2_health_rgb/PlantVillage', 'data/raw/vision/PlantVillage')
    with open(api_path, 'w', encoding='utf-8') as f:
        f.write(api_data)

# 6. Fix web_inference.py
web_path = PROJECT_ROOT / "src" / "api" / "web_inference.py"
if web_path.exists():
    with open(web_path, 'r', encoding='utf-8') as f:
        web_data = f.read()
    web_data = web_data.replace('data/layer2_health_rgb/PlantVillage', 'data/raw/vision/PlantVillage')
    with open(web_path, 'w', encoding='utf-8') as f:
        f.write(web_data)

print("All occurrences of 'layer2_health_rgb' have been replaced with 'raw/vision'.")
