import os
import argparse
import importlib.util
import sys
from pathlib import Path
from datetime import datetime

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
# Resolves accurately whether currently in /evaluation/ or moved to /src/evaluation/
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description='Run CNN eval, RF eval, then summarise results')
    parser.add_argument('--cnn_model', default=str(PROJECT_ROOT / 'models/demeter_cnn.keras'))
    parser.add_argument('--baseline_cnn_model', default=None, help='Optional baseline CNN model')
    parser.add_argument('--cnn_test_dir', required=True)
    parser.add_argument('--cnn_mode', choices=['full', 'simple'], default='full')
    parser.add_argument('--rf_model', default=str(PROJECT_ROOT / 'models/demeter_rf.joblib'))
    parser.add_argument('--rf_csv', required=True)
    parser.add_argument('--rf_mode', choices=['full', 'simple'], default='full')
    parser.add_argument('--kmeans_model', default=None, help='Path to K-Means model')
    parser.add_argument('--kmeans_csv', default=None, help='Path to K-Means dataset')
    parser.add_argument('--out_base', default=str(PROJECT_ROOT / 'evaluation_outputs'))
    parser.add_argument('--run_name', default=None)
    args = parser.parse_args()

    root = os.path.dirname(__file__)
    cnn_path = os.path.join(root, 'evaluate_cnn.py')
    rf_path = os.path.join(root, 'evaluate_rf.py')
    summary_path = os.path.join(root, 'summarise_results.py')
    kmeans_path = os.path.join(root, 'evaluate_kmeans.py')

    print('Loading evaluation modules...')
    cnn_mod = load_module_from_path('eval_cnn', cnn_path)
    rf_mod = load_module_from_path('eval_rf', rf_path)
    sum_mod = load_module_from_path('eval_sum', summary_path)
    
    if args.kmeans_model:
        kmeans_mod = load_module_from_path('eval_kmeans', kmeans_path)

    if args.run_name:
        active_out_base = os.path.join(args.out_base, args.run_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        active_out_base = os.path.join(args.out_base, f"run_{timestamp}")

    cnn_out = os.path.join(active_out_base, 'cnn' if args.cnn_mode == 'full' else 'cnn_simple')
    rf_out = os.path.join(active_out_base, 'rf' if args.rf_mode == 'full' else 'rf_simple')

    try:
        print(f"Running Primary CNN ({args.cnn_mode}) -> output: {cnn_out}/primary")
        out_primary = os.path.join(cnn_out, 'primary')
        if args.cnn_mode == 'full':
            cnn_mod.evaluate_cnn(args.cnn_model, args.cnn_test_dir, out_dir=out_primary)
        else:
            cnn_mod.evaluate_cnn_simple(args.cnn_model, args.cnn_test_dir, out_dir=out_primary)
            
        if args.baseline_cnn_model and os.path.exists(args.baseline_cnn_model):
            print(f"Running Baseline CNN ({args.cnn_mode}) -> output: {cnn_out}/baseline")
            out_baseline = os.path.join(cnn_out, 'baseline')
            if args.cnn_mode == 'full':
                cnn_mod.evaluate_cnn(args.baseline_cnn_model, args.cnn_test_dir, out_dir=out_baseline)
            else:
                cnn_mod.evaluate_cnn_simple(args.baseline_cnn_model, args.cnn_test_dir, out_dir=out_baseline)

        print(f"Running RF ({args.rf_mode}) -> output: {rf_out}")
        if args.rf_mode == 'full':
            rf_mod.evaluate_rf(args.rf_model, args.rf_csv, out_dir=rf_out)
        else:
            rf_mod.evaluate_rf_simple(args.rf_model, args.rf_csv, out_dir=rf_out)
            
        if args.kmeans_model and args.kmeans_csv and os.path.exists(args.kmeans_model):
            print(f"Running K-Means Evaluation -> output: {active_out_base}/kmeans")
            kmeans_mod.evaluate_kmeans(args.kmeans_model, args.kmeans_csv, out_dir=os.path.join(active_out_base, 'kmeans'))

        print('Summarising results...')
        sum_mod.summarise(cnn_out=cnn_out, rf_out=rf_out, out_file=os.path.join(active_out_base, 'summary.csv'), run_name=args.run_name)

        print('Evaluation suite finished. Summary at', os.path.join(active_out_base, 'summary.csv'))
    except Exception as e:
        print('Evaluation suite failed with error:', e)
        raise

if __name__ == '__main__':
    main()
