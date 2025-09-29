#!/usr/bin/env python3
"""
Test runner script for the E-commerce AI Chatbot project.
Provides convenient commands to run different types of tests.
"""

import subprocess
import sys
import argparse


def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print('='*50)

    try:
        result = subprocess.run(command, capture_output=False, text=True)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTest run interrupted by user.")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for E-commerce AI Chatbot")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    parser.add_argument("--integration", action="store_true", help="Include integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Base pytest command
    cmd = ["pytest"]

    if args.verbose:
        cmd.append("-v")

    if not args.integration:
        # Skip integration tests by default
        cmd.extend(["-m", "not integration"])

    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])

    # Run the tests
    success = run_command(cmd, "Running tests")

    if success:
        print(f"\n{'='*50}")
        print("✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
        print('='*50)
    else:
        print(f"\n{'='*50}")
        print("❌ Some tests failed. Check the output above for details.")
        print('='*50)
        sys.exit(1)


if __name__ == "__main__":
    main()