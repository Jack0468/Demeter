import argparse
import sys
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description="Demeter Plant Health Management System")
    parser.add_argument('--mode', choices=['cli', 'dashboard', 'both'], default='both',
                        help="Run mode: 'cli' (terminal output + CSV), 'dashboard' (JSON for web API), or 'both'")
    return parser.parse_args()

args = parse_args()
print(f"Mode: {args.mode}")
