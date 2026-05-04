import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Demeter Plant Health Management System")
    parser.add_argument('--mode', choices=['cli', 'dashboard', 'both'], default='both',
                        help="Run mode: 'cli' (terminal output + CSV), 'dashboard' (JSON for web API), or 'both'")
    return parser.parse_args()

args = parse_args()
print(f"Mode: {args.mode}")
