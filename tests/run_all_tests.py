"""
run_all_tests.py

Comprehensive test runner for all unit tests and model evaluation.
Generates detailed report on test results.
"""

import os
import sys
import unittest
import json
from datetime import datetime
from io import StringIO

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))


def run_unit_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS")
    print("="*70)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    test_modules = [
        'test_output_formatter',
        'test_status_engine',
        # 'test_api_server'  # May skip if API dependencies not fully configured
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"✓ Loaded {module_name}")
        except ImportError as e:
            print(f"✗ Could not load {module_name}: {e}")
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return {
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
        "failure_details": [
            {
                "test": str(test),
                "traceback": traceback
            } for test, traceback in result.failures
        ],
        "error_details": [
            {
                "test": str(test),
                "traceback": traceback
            } for test, traceback in result.errors
        ]
    }


def run_model_evaluation():
    """Run model evaluation."""
    print("\n" + "="*70)
    print("RUNNING MODEL EVALUATION")
    print("="*70)
    
    # Import and run model evaluation
    try:
        from model_evaluation_detailed import ModelEvaluator
        
        evaluator = ModelEvaluator()
        evaluator.check_model_availability()
        evaluator.evaluate_cnn_plantvillage()
        evaluator.evaluate_rf_danforth()
        evaluator.evaluate_data_availability()
        evaluator.evaluate_inference_engine()
        evaluator.evaluate_output_formatter()
        evaluator.evaluate_status_engine()
        
        print("\n✓ Model evaluation completed")
        return evaluator.results
        
    except Exception as e:
        print(f"\n✗ Error running model evaluation: {e}")
        return {"error": str(e)}


def generate_final_report(test_results, eval_results):
    """Generate comprehensive final report."""
    print("\n" + "="*70)
    print("FINAL TEST REPORT")
    print("="*70)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_results": test_results,
        "evaluation_results": eval_results,
        "summary": {}
    }
    
    # Test summary
    print("\n📋 UNIT TEST SUMMARY:")
    print(f"  Total Tests: {test_results['total_tests']}")
    print(f"  Passed: {test_results['total_tests'] - test_results['failures'] - test_results['errors']}")
    print(f"  Failed: {test_results['failures']}")
    print(f"  Errors: {test_results['errors']}")
    print(f"  Status: {'✓ PASS' if test_results['success'] else '✗ FAIL'}")
    
    if test_results['failures'] > 0:
        print(f"\n  Failed Tests:")
        for failure in test_results['failure_details'][:3]:  # Show first 3
            print(f"    - {failure['test']}")
    
    if test_results['errors'] > 0:
        print(f"\n  Test Errors:")
        for error in test_results['error_details'][:3]:  # Show first 3
            print(f"    - {error['test']}")
    
    # Evaluation summary
    print("\n📊 MODEL EVALUATION SUMMARY:")
    
    if 'model_availability' in eval_results:
        models = eval_results['model_availability']
        available = sum(1 for m in models.values() if m.get('available'))
        print(f"  Models Available: {available}/{len(models)}")
        for name, model in models.items():
            status = "✓" if model['available'] else "✗"
            print(f"    {status} {name}")
    
    if 'cnn_plantvillage' in eval_results:
        cnn = eval_results['cnn_plantvillage']
        if cnn.get('status') == 'loaded':
            print(f"\n  CNN PlantVillage: ✓ LOADED")
            print(f"    Layers: {cnn.get('layers')}")
            print(f"    Parameters: {cnn.get('total_params'):,}")
        else:
            print(f"\n  CNN PlantVillage: ✗ {cnn.get('status')}")
    
    if 'rf_danforth' in eval_results:
        rf = eval_results['rf_danforth']
        if rf.get('status') == 'loaded':
            print(f"\n  RF Danforth: ✓ LOADED")
            print(f"    Type: {rf.get('type')}")
            print(f"    Trees: {rf.get('n_trees')}")
        else:
            print(f"\n  RF Danforth: ✗ {rf.get('status')}")
    
    # Module status
    print("\n🔧 MODULE STATUS:")
    modules_status = {
        "inference_engine": eval_results.get('inference_engine', {}).get('status'),
        "output_formatter": eval_results.get('output_formatter', {}).get('status'),
        "status_engine": eval_results.get('status_engine', {}).get('status')
    }
    for module, status in modules_status.items():
        icon = "✓" if status in ['available', 'functional'] else "✗"
        print(f"  {icon} {module}: {status}")
    
    # Overall status
    overall_pass = test_results['success'] and all(
        s in ['available', 'functional'] for s in modules_status.values()
    )
    
    print("\n" + "="*70)
    if overall_pass:
        print("✓ ALL SYSTEMS OPERATIONAL")
    else:
        print("⚠ SOME ISSUES DETECTED - See above for details")
    print("="*70)
    
    # Save full report
    os.makedirs("data/outputs", exist_ok=True)
    report_file = "data/outputs/test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Full report saved to {report_file}")
    
    report['summary'] = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "test_pass": test_results['success'],
        "models_available": modules_status
    }
    
    return report


def main():
    """Run all tests and evaluations."""
    print("\n" + "🌱 "*10)
    print("DEMETER - COMPREHENSIVE TEST & EVALUATION SUITE")
    print("🌱 "*10)
    print(f"\nStarted: {datetime.now().isoformat()}")
    
    # Run tests
    print("\n[1/2] Running unit tests...")
    test_results = run_unit_tests()
    
    # Run model evaluation
    print("\n[2/2] Running model evaluation...")
    eval_results = run_model_evaluation()
    
    # Generate report
    report = generate_final_report(test_results, eval_results)
    
    print(f"\nCompleted: {datetime.now().isoformat()}")
    
    # Exit with appropriate code
    sys.exit(0 if report['summary']['overall_status'] == 'PASS' else 1)


if __name__ == "__main__":
    main()
