#!/usr/bin/env python3
"""
Automation script for the complete SAT solving and verification workflow.

This script automates the following steps:
1. Generate CNF constraints
2. Convert graphs to canonical form
3. Apply mapping to vectors
4. Run cadical solver
5. Run drat-trim
6. Verify DRAT proof

Usage:
    python3 automate_workflow.py --n 28 --block 0 --lex-option no-lex \\
        --graphs sic25_extension.txt --output-dir order28_results \\
        --partition 25 --vectors-file cadical-rcl/data/SI-C-c2-labeled-853-25.lad \\
        [--graph-index 1] [--skip-generation] [--skip-canonical] [--skip-verify]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def get_base_cnf_name(n, block, lex_option):
    """Generate base CNF filename from parameters."""
    base_name = f"constraints_{n}_{block}"
    
    if lex_option == "lex-greatest":
        return base_name + "_lex_greatest"
    elif lex_option == "no-lex":
        return base_name + "_no_lex"
    else:  # lex-least is the default
        return base_name + "_1"


def step1_generate_cnf(n, block, lex_option, skip_if_exists=True):
    """Step 1: Generate CNF constraints."""
    print("=" * 80)
    print("STEP 1: Generating CNF constraints")
    print("=" * 80)
    
    base_cnf = get_base_cnf_name(n, block, lex_option)
    
    if skip_if_exists and os.path.exists(base_cnf):
        print(f"✓ Base CNF file '{base_cnf}' already exists, skipping generation")
        return True, base_cnf
    
    cmd = ["python3", "gen_instance/generate.py", str(n), str(block), lex_option]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully generated base CNF: {base_cnf}")
        return True, base_cnf
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating CNF: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False, None


def step2_convert_graphs(graphs_file, base_cnf, output_dir, color_representation=None):
    """Step 2: Convert graphs to canonical form."""
    print("\n" + "=" * 80)
    print("STEP 2: Converting graphs to canonical form")
    print("=" * 80)
    
    if not os.path.exists(graphs_file):
        print(f"✗ Graphs file '{graphs_file}' not found")
        return False
    
    if not os.path.exists(base_cnf):
        print(f"✗ Base CNF file '{base_cnf}' not found")
        return False
    
    cmd = ["python3", "simple_graph6_to_canonical.py",
           "--graphs", graphs_file,
           "--base_cnf", base_cnf,
           "--output_dir", output_dir]
    
    if color_representation:
        cmd.extend(["--color", color_representation])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully converted graphs to canonical form")
        print(f"  Output directory: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting graphs: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def step3_apply_mapping(output_dir, graph_index, vectors_file):
    """Step 3: Apply mapping to vectors."""
    print("\n" + "=" * 80)
    print(f"STEP 3: Applying mapping to vectors for graph_{graph_index:03d}")
    print("=" * 80)
    
    mapping_file = os.path.join(output_dir, f"graph_{graph_index:03d}_mapping.txt")
    mapped_vectors_file = os.path.join(output_dir, f"graph_{graph_index:03d}_mapped_vectors")
    
    if not os.path.exists(mapping_file):
        print(f"✗ Mapping file '{mapping_file}' not found")
        return False, None
    
    # Check if vectors_file exists, if not try per-graph vectors file or .vec extension
    if not os.path.exists(vectors_file):
        # Try per-graph vectors file
        per_graph_vectors = os.path.join(output_dir, f"graph_{graph_index:03d}_vectors.txt")
        if os.path.exists(per_graph_vectors):
            print(f"  Using per-graph vectors file: {per_graph_vectors}")
            vectors_file = per_graph_vectors
        else:
            # Try .vec extension (for Python dictionary format with sqrt expressions)
            vec_file = vectors_file if vectors_file.endswith('.vec') else vectors_file + '.vec'
            if os.path.exists(vec_file):
                print(f"  Using .vec file: {vec_file}")
                vectors_file = vec_file
            else:
                print(f"✗ Vectors file '{vectors_file}' not found")
                print(f"  Also tried: {per_graph_vectors}, {vec_file}")
                return False, None
    
    cmd = ["python3", "apply_mapping_to_vectors.py",
           "--mapping", mapping_file,
           "--vectors", vectors_file,
           "--output", mapped_vectors_file]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully applied mapping to vectors")
        print(f"  Output: {mapped_vectors_file}")
        return True, mapped_vectors_file
    except subprocess.CalledProcessError as e:
        print(f"✗ Error applying mapping: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False, None


def step4_run_cadical(output_dir, graph_index, n, partition, ortho, complex_mode, vectors_file):
    """Step 4: Run cadical solver."""
    print("\n" + "=" * 80)
    print(f"STEP 4: Running cadical solver for graph_{graph_index:03d}")
    print("=" * 80)
    
    cnf_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.cnf")
    
    if not os.path.exists(cnf_file):
        print(f"✗ CNF file '{cnf_file}' not found")
        return False
    
    if vectors_file and not os.path.exists(vectors_file):
        print(f"✗ Vectors file '{vectors_file}' not found")
        return False
    
    cmd = ["./cadical-rcl/build/cadical", cnf_file,
           "--order", str(n)]
    
    if partition:
        cmd.extend(["--partition", str(partition)])
    
    if ortho:
        cmd.append("--ortho")
    
    if complex_mode:
        cmd.append("--complex")
    
    if vectors_file:
        cmd.extend(["--vectors-file", vectors_file])
    
    print(f"Running: {' '.join(cmd)}")
    print()  # Empty line before cadical output
    
    # Run cadical and output directly to terminal (not captured)
    result = subprocess.run(cmd)
    
    print()  # Empty line after cadical output
    
    # CaDiCaL exit codes (SAT competition standard):
    #   0 = normal termination / UNKNOWN
    #  10 = SATISFIABLE - problem has a solution
    #  20 = UNSATISFIABLE - problem has no solution (proof generated)
    #  Other = error
    
    if result.returncode == 10:
        print(f"⚠ cadical found SAT (satisfiable) - problem has a solution")
        print(f"  Note: This may not be the expected result if you're trying to prove unsatisfiability")
    elif result.returncode == 20:
        print(f"✓ Successfully ran cadical solver (UNSAT - problem is unsatisfiable)")
    elif result.returncode == 0:
        print(f"✓ Successfully ran cadical solver (normal termination)")
    else:
        print(f"✗ Error running cadical: exit code {result.returncode}")
        return False
    
    # Check for output files
    drat_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.drat")
    perm_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.perm")
    ortho_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.ortho")
    
    files_created = []
    if os.path.exists(drat_file):
        files_created.append(f"  DRAT: {drat_file}")
    if os.path.exists(perm_file):
        files_created.append(f"  Perm: {perm_file}")
    if os.path.exists(ortho_file):
        files_created.append(f"  Ortho: {ortho_file}")
    
    if files_created:
        print("  Generated files:")
        for f in files_created:
            print(f)
    
    # For SAT results (exit code 10), DRAT file might not exist
    # Only proceed with verification if we have the necessary files
    if result.returncode == 10:
        if not os.path.exists(drat_file):
            print(f"  Note: No DRAT file generated for SAT result (this is expected)")
            # Return True but note that verification will be skipped
            return True
    
    return True


def step5_run_drat_trim(output_dir, graph_index):
    """Step 5: Run drat-trim."""
    print("\n" + "=" * 80)
    print(f"STEP 5: Running drat-trim for graph_{graph_index:03d}")
    print("=" * 80)
    
    cnf_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.cnf")
    drat_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.drat")
    
    if not os.path.exists(cnf_file):
        print(f"✗ CNF file '{cnf_file}' not found")
        return False
    
    if not os.path.exists(drat_file):
        print(f"⚠ DRAT file '{drat_file}' not found")
        print(f"  This is expected if cadical returned SAT (exit code 10)")
        print(f"  Skipping drat-trim (no proof to verify)")
        return True  # Not an error, just skip this step
    
    cmd = ["./drat-trim/drat-trim", cnf_file, drat_file]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully ran drat-trim")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running drat-trim: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def step6_verify_drat(output_dir, graph_index, ortho, complex_mode, verbose=False):
    """Step 6: Verify DRAT proof."""
    print("\n" + "=" * 80)
    print(f"STEP 6: Verifying DRAT proof for graph_{graph_index:03d}")
    print("=" * 80)
    
    perm_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.perm")
    drat_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.drat")
    vars_file = os.path.join(output_dir, f"graph_{graph_index:03d}_vars.txt")
    ortho_file = os.path.join(output_dir, f"graph_{graph_index:03d}_with_vars.ortho")
    
    if not os.path.exists(perm_file):
        print(f"⚠ Permutation file '{perm_file}' not found")
        print(f"  This is expected if cadical returned SAT (exit code 10)")
        print(f"  Skipping DRAT verification (no proof to verify)")
        return True  # Not an error, just skip this step
    
    if not os.path.exists(drat_file):
        print(f"⚠ DRAT file '{drat_file}' not found")
        print(f"  This is expected if cadical returned SAT (exit code 10)")
        print(f"  Skipping DRAT verification (no proof to verify)")
        return True  # Not an error, just skip this step
    
    if not os.path.exists(vars_file):
        print(f"✗ Variables file '{vars_file}' not found")
        return False
    
    cmd = ["python", "drat_verifier.py", perm_file, drat_file,
           "--fixed-edges", vars_file]
    
    if ortho and os.path.exists(ortho_file):
        cmd.extend(["--ortho", ortho_file])
    
    if complex_mode:
        cmd.append("--complex")
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ DRAT verification passed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ DRAT verification failed: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def get_graph_count(graphs_file):
    """Count the number of graphs in the graphs file."""
    try:
        with open(graphs_file, 'r') as f:
            return len([line for line in f if line.strip()])
    except Exception as e:
        print(f"Error reading graphs file: {e}")
        return 0


def process_single_graph(args, graph_index):
    """Process a single graph through all steps."""
    print(f"\n{'#' * 80}")
    print(f"Processing graph_{graph_index:03d}")
    print(f"{'#' * 80}")
    
    # Step 3: Apply mapping to vectors
    success, mapped_vectors = step3_apply_mapping(
        args.output_dir, graph_index, args.vectors_file
    )
    if not success:
        return False
    
    # Step 4: Run cadical
    success = step4_run_cadical(
        args.output_dir, graph_index, args.n, args.partition,
        args.ortho, args.complex, mapped_vectors
    )
    if not success:
        return False
    
    # Step 5: Run drat-trim
    # COMMENTED OUT: Script ends after cadical for now
    # if not args.skip_drat_trim:
    #     success = step5_run_drat_trim(args.output_dir, graph_index)
    #     if not success:
    #         return False
    
    # Step 6: Verify DRAT
    # COMMENTED OUT: Script ends after cadical for now
    # if not args.skip_verify:
    #     success = step6_verify_drat(
    #         args.output_dir, graph_index, args.ortho, args.complex, args.verbose
    #     )
    #     if not success:
    #         return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Automate the complete SAT solving and verification workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single graph:
  python3 automate_workflow.py --n 28 --block 0 --lex-option no-lex \\
      --graphs sic25_extension.txt --output-dir order28_results \\
      --partition 25 --vectors-file cadical-rcl/data/SI-C-c2-labeled-853-25.lad --graph-index 1

  # Process all graphs:
  python3 automate_workflow.py --n 28 --block 0 --lex-option no-lex \\
      --graphs sic25_extension.txt --output-dir order28_results \\
      --partition 25 --vectors-file cadical-rcl/data/SI-C-c2-labeled-853-25.lad

  # Skip generation if already done:
  python3 automate_workflow.py --n 28 --block 0 --lex-option no-lex \\
      --graphs sic25_extension.txt --output-dir order28_results \\
      --partition 25 --vectors-file cadical-rcl/data/SI-C-c2-labeled-853-25.lad \\
      --skip-generation --skip-canonical
        """
    )
    
    # Core parameters
    parser.add_argument('--n', type=int, required=True,
                       help='Graph order/size (e.g., 25)')
    parser.add_argument('--block', type=str, required=True,
                       help='Block identifier/color ratio (e.g., "0.1")')
    parser.add_argument('--lex-option', type=str, required=True,
                       choices=['lex-least', 'lex-greatest', 'no-lex'],
                       help='Isomorphism blocking option')
    parser.add_argument('--graphs', type=str, required=True,
                       help='Path to file containing graph6 strings')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory name')
    parser.add_argument('--partition', type=int, required=True,
                       help='Partition value for cadical solver')
    parser.add_argument('--vectors-file', type=str, required=True,
                       help='Path to vectors file (or directory containing vectors)')
    
    # Graph selection
    parser.add_argument('--graph-index', type=int, default=None,
                       help='Process specific graph by index (1-based). If not specified, process all graphs')
    
    # Flags
    parser.add_argument('--ortho', action='store_true', default=True,
                       help='Enable orthogonality mode (default: True)')
    parser.add_argument('--no-ortho', dest='ortho', action='store_false',
                       help='Disable orthogonality mode')
    parser.add_argument('--complex', action='store_true', default=True,
                       help='Enable complex mode (default: True)')
    parser.add_argument('--no-complex', dest='complex', action='store_false',
                       help='Disable complex mode')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output for verification')
    
    # Optional parameters
    parser.add_argument('--color-representation', type=str, default=None,
                       help='Color representation for RCL-color')
    
    # Skip options
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip CNF generation step (use existing file)')
    parser.add_argument('--skip-canonical', action='store_true',
                       help='Skip canonical conversion step (use existing files)')
    parser.add_argument('--skip-drat-trim', action='store_true',
                       help='Skip drat-trim step')
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip DRAT verification step')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.graphs):
        print(f"Error: Graphs file '{args.graphs}' not found")
        sys.exit(1)
    
    # Step 1: Generate CNF
    if not args.skip_generation:
        success, base_cnf = step1_generate_cnf(args.n, args.block, args.lex_option)
        if not success:
            print("\n✗ Workflow failed at Step 1")
            sys.exit(1)
    else:
        base_cnf = get_base_cnf_name(args.n, args.block, args.lex_option)
        if not os.path.exists(base_cnf):
            print(f"Error: Base CNF file '{base_cnf}' not found (use --skip-generation only if file exists)")
            sys.exit(1)
        print(f"✓ Using existing base CNF: {base_cnf}")
    
    # Step 2: Convert graphs to canonical form
    if not args.skip_canonical:
        success = step2_convert_graphs(
            args.graphs, base_cnf, args.output_dir, args.color_representation
        )
        if not success:
            print("\n✗ Workflow failed at Step 2")
            sys.exit(1)
    else:
        if not os.path.exists(args.output_dir):
            print(f"Error: Output directory '{args.output_dir}' not found (use --skip-canonical only if directory exists)")
            sys.exit(1)
        print(f"✓ Using existing output directory: {args.output_dir}")
    
    # Determine which graphs to process
    if args.graph_index is not None:
        graph_indices = [args.graph_index]
    else:
        # Process all graphs
        graph_count = get_graph_count(args.graphs)
        if graph_count == 0:
            print("Error: No graphs found in graphs file")
            sys.exit(1)
        graph_indices = list(range(1, graph_count + 1))
        print(f"\nProcessing {graph_count} graph(s)")
    
    # Process each graph
    success_count = 0
    for graph_index in graph_indices:
        success = process_single_graph(args, graph_index)
        if success:
            success_count += 1
        else:
            print(f"\n✗ Failed to process graph_{graph_index:03d}")
            if args.graph_index is not None:
                # If processing single graph, exit on failure
                sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {success_count}/{len(graph_indices)} graph(s)")
    
    if success_count == len(graph_indices):
        print("✓ All graphs processed successfully!")
        sys.exit(0)
    else:
        print(f"✗ {len(graph_indices) - success_count} graph(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

