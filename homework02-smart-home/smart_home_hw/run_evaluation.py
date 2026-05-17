"""
Entry point for evaluating RequestSolver strategies on the homework test set.

Usage:
    python run_evaluation.py

This script:
    1. Loads requests and preferences from ../requests.json and ../preferences.json
    2. Instantiates EnvironmentManagerAgent and RequestSolverAgent
    3. For each request: solve, evaluate, accumulate metrics
    4. Prints aggregated EvaluationMetrics
"""

import json
import sys
import time
import os
from pathlib import Path

from models import Request, ActionOutput, Preference, TimeInterval
from environment_manager import EnvironmentManagerAgent
from request_solver import RequestSolverAgent, DummyRequestSolver
from evaluation import evaluate_single, EvaluationMetrics
from llm_client import get_llm_client

from request_solver import FullContextSolver, SequentialSolver, SemanticSolver


SIMULATOR_URL = "http://localhost:8080"
HOMEWORK_DIR = Path(__file__).parent.parent
VERBOSE = os.getenv("VERBOSE_AGENTS", "").lower() in ("1", "true", "yes")
AGENT_STARTUP_DELAY = 0.5  # Give agent thread time to start


def load_requests(filepath: Path) -> list[Request]:
    """Load requests from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    requests = []
    for entry in data:
        output = [ActionOutput(**out) for out in entry.get("output", [])]
        req = Request(
            id=entry["id"],
            issued_at=entry.get("issued_at", "12:00"),
            input=entry["input"],
            output=output,
        )
        requests.append(req)
    return requests


def load_preferences(filepath: Path) -> list[Preference]:
    """Load preferences from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    preferences = []
    for entry in data:
        pref = Preference(
            device_type=entry["device_type"],
            room=entry["room"],
            dislike_interval=TimeInterval(**entry["dislike_interval"]),
            reason=entry.get("reason", ""),
        )
        preferences.append(pref)
    return preferences


def main():
    """Run evaluation."""
    print("=" * 70)
    print("SmartHome Homework Evaluation")
    print("=" * 70)

    # Load test data
    requests_file = HOMEWORK_DIR / "requests.json"
    prefs_file = HOMEWORK_DIR / "preferences.json"

    if not requests_file.exists() or not prefs_file.exists():
        print(f"Error: Test files not found")
        print(f"  Expected: {requests_file}")
        print(f"  Expected: {prefs_file}")
        sys.exit(1)

    requests = load_requests(requests_file)
    preferences = load_preferences(prefs_file)

    print(f"\nLoaded {len(requests)} requests")
    print(f"Loaded {len(preferences)} preferences")
    print(f"Simulator URL: {SIMULATOR_URL}")

    # Initialize agents
    try:
        llm_client = get_llm_client()
    except ValueError as e:
        print(f"Error: {e}")
        print("Set OPENAI_API_KEY in .env file (or use dummy solver for testing)")
        llm_client = None

    # Start EnvironmentManager agent (runs in separate thread)
    env_manager = EnvironmentManagerAgent(SIMULATOR_URL, preferences, verbose=VERBOSE)
    env_manager.start()
    print("✓ EnvironmentManager agent started (message handler thread running)")

    # Give the agent thread time to fully initialize
    time.sleep(AGENT_STARTUP_DELAY)

    # Create RequestSolver (uses agent communication protocol)
    solver = FullContextSolver(env_manager, llm_client, verbose=VERBOSE)
    print(f"✓ {type(solver).__name__} created")

    if VERBOSE:
        print("✓ Verbose agent communication logging enabled")
    else:
        print("  Tip: Use 'VERBOSE_AGENTS=1 python run_evaluation.py' to see agent communication details")

    # Run evaluation
    metrics = EvaluationMetrics()
    results = []

    print("\n" + "=" * 70)
    print("Running evaluation...")
    print("=" * 70)

    for req_idx, request in enumerate(requests):
        # Solve using agent communication protocol
        start = time.time()
        try:
            predicted = solver.solve(request)
            duration = time.time() - start
        except NotImplementedError:
            print(f"\n[{req_idx + 1}/{len(requests)}] {request.id}")
            print("  ERROR: RequestSolverAgent.solve() not implemented")
            env_manager.stop()
            sys.exit(1)
        except TimeoutError as e:
            predicted = []
            duration = time.time() - start
            print(f"\n[{req_idx + 1}/{len(requests)}] {request.id}")
            print(f"  ERROR: Agent communication timeout: {e}")
            print("  (EnvironmentManager may not be responding)")
        except Exception as e:
            predicted = []
            duration = time.time() - start
            print(f"\n[{req_idx + 1}/{len(requests)}] {request.id}")
            print(f"  ERROR: {type(e).__name__}: {e}")

        # Evaluate (executes actions at simulator, verifies results, resets home)
        result = evaluate_single(request, predicted, SIMULATOR_URL, timeout=30)
        result.duration_seconds = duration
        results.append(result)

        # Accumulate metrics
        metrics.total_tests += 1

        if result.success == "True":
            metrics.successful_tests += 1
        elif result.success == "Quantifiable":
            metrics.quantifiable_tests += 1
        else:
            metrics.failed_tests += 1

        metrics.total_expected_actions += len(result.expected_actions)
        metrics.total_matched_actions += len(result.matched_actions)
        metrics.total_missing_actions += len(result.missing_actions)
        metrics.total_extra_actions += len(result.extra_actions)
        metrics.total_properties_checked += result.properties_checked
        metrics.total_properties_matched += result.properties_matched
        metrics.total_expected_impossible += result.expected_impossible
        metrics.total_detected_impossible += getattr(result, "total_detected_impossible", 0) # FIXED
        metrics.total_duration += duration

        # Progress - print every request
        print(f"[{req_idx + 1}/{len(requests)}] {request.id:30} — {result.success}")

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(json.dumps(metrics.to_dict(), indent=2))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Success Rate: {metrics.success_rate:.1%}")
    print(f"Quantifiable Rate: {metrics.quantifiable_rate:.1%}")
    print(f"Success + Quantifiable: {metrics.success_or_quantifiable_rate:.1%}")
    print(f"Action F1: {metrics.action_f1:.1%}")
    print(f"Property Accuracy: {metrics.property_accuracy:.1%}")
    print(f"Impossible Detection Rate: {metrics.impossible_detection_rate:.1%}")
    print(f"Average Duration: {metrics.total_duration / metrics.total_tests:.2f}s")

    # Cleanup
    print("\n" + "=" * 70)
    print("Shutting down agents...")
    print("=" * 70)
    env_manager.stop()
    print("✓ EnvironmentManager agent stopped")

    os.makedirs(HOMEWORK_DIR / "results", exist_ok=True)
    results_path = HOMEWORK_DIR / "results" / f"{type(solver).__name__}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": metrics.to_dict(),
            "results": [result.to_dict() for result in results]
        }, f)
        
    print(f"✓ Results saved to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
