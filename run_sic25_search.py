#!/usr/bin/env python3
"""
Simple push-button script to run the complete 25-ray SI-C extension workflow.

This script:
1. Generates base CNF constraints for target order
2. Adds 25-ray SI-C unit clauses from sic-25.vars
3. Runs CaDiCaL-RCL solver with correct configuration
4. Reports results

Usage:
    python3 run_sic25_search.py [--order N] [--output-dir DIR]

Examples:
    # Search for order 28 KS sets extending 25-ray SI-C
    python3 run_sic25_search.py --order 28
    
    # Search for order 33 (full paper experiment)
    python3 run_sic25_search.py --order 33
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)


def step1_generate_base_cnf(order, skip_if_exists=True):
    """Generate base CNF constraints."""
    print_header(f"STEP 1: Generating base CNF for order {order}")
    
    # The generate.py script creates the file in the CURRENT directory (not gen_instance/)
    base_cnf = f"constraints_{order}_0_no_lex"
    
    if skip_if_exists and Path(base_cnf).exists():
        print(f"✓ Base CNF '{base_cnf}' already exists, skipping generation")
        return True, base_cnf
    
    # Run generation from gen_instance directory, but file is created in current dir
    cmd = ["python3", "gen_instance/generate.py", str(order), "0", "no-lex"]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Verify file was created in current directory
        if not Path(base_cnf).exists():
            print(f"✗ Expected file not created: {base_cnf}")
            return False, None
            
        print(f"✓ Successfully generated: {base_cnf}")
        return True, base_cnf
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating CNF: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False, None


def step2_add_sic25_constraints(base_cnf, output_dir):
    """Add 25-ray SI-C unit clauses to base CNF."""
    print_header("STEP 2: Adding 25-ray SI-C constraints")
    
    vars_file = "sic-25.vars"
    if not Path(vars_file).exists():
        print(f"✗ Variables file '{vars_file}' not found")
        return False, None
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output CNF with SI-C constraints
    output_cnf = output_path / f"order_{Path(base_cnf).stem}_sic25.cnf"
    
    cmd = ["python3", "utils/add_vars_to_cnf.py", base_cnf, vars_file, str(output_cnf)]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully added SI-C constraints")
        print(f"  Input: {base_cnf}")
        print(f"  Variables: {vars_file}")
        print(f"  Output: {output_cnf}")
        return True, str(output_cnf)
    except subprocess.CalledProcessError as e:
        print(f"✗ Error adding variables: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False, None


def step3_run_cadical(cnf_file, order, output_dir, partition=25, complex_mode=False):
    """Run CaDiCaL-RCL solver."""
    print_header("STEP 3: Running CaDiCaL-RCL solver")
    
    vectors_file = "sic-25-vectors.txt"
    
    if not Path(vectors_file).exists():
        print(f"✗ Vectors file '{vectors_file}' not found")
        return False
    
    # Construct cadical command
    cmd = [
        "./cadical-rcl/build/cadical",
        cnf_file,
        "--order", str(order),
        "--partition", str(partition),
        "--vectors-file", vectors_file,
        "--unembeddable-check", "0",
        "--ortho"
    ]
    
    if complex_mode:
        cmd.append("--complex")
    
    # Add output file for logging
    log_file = Path(output_dir) / f"order_{order}_sic25.log"
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    
    try:
        with open(log_file, 'w') as log:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Write to log and display
            output = result.stdout
            log.write(output)
            print(output)
        
        if result.returncode == 20:
            print(f"\n✓ Result: UNSATISFIABLE")
            print(f"  No KS sets of order {order} extend the 25-ray SI-C set")
        elif result.returncode == 10:
            print(f"\n✓ Result: SATISFIABLE")
            print(f"  Found KS set(s) extending the 25-ray SI-C set")
        else:
            print(f"\n⚠ Solver exited with code {result.returncode}")
        
        print(f"\n  Log file: {log_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error running cadical: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run exhaustive search for KS sets extending 25-ray SI-C',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for order 28 KS sets
  python3 run_sic25_search.py --order 28
  
  # Search for order 33 (full paper experiment)
  python3 run_sic25_search.py --order 33
  
  # Search with complex arithmetic
  python3 run_sic25_search.py --order 30 --complex
        """
    )
    
    parser.add_argument('--order', type=int, default=28,
                       help='Target graph order (default: 28)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--partition', type=int, default=25,
                       help='Starting partition size (default: 25 for SI-C)')
    parser.add_argument('--complex', action='store_true',
                       help='Use complex arithmetic')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip CNF generation if file exists')
    
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║  SAT + nauty: 25-ray SI-C Extension Search                                ║
║  Target Order: {args.order:<2}                                                        ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Generate base CNF
    success, base_cnf = step1_generate_base_cnf(args.order, args.skip_generation)
    if not success:
        print("\n✗ Workflow failed at step 1")
        sys.exit(1)
    
    # Step 2: Add SI-C constraints
    success, cnf_with_sic = step2_add_sic25_constraints(base_cnf, args.output_dir)
    if not success:
        print("\n✗ Workflow failed at step 2")
        sys.exit(1)
    
    # Step 3: Run CaDiCaL
    success = step3_run_cadical(
        cnf_with_sic, 
        args.order, 
        args.output_dir,
        args.partition,
        args.complex
    )
    if not success:
        print("\n✗ Workflow failed at step 3")
        sys.exit(1)
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║  Workflow Complete                                                         ║
║  Results saved to: {args.output_dir:<54} ║
╚════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()

