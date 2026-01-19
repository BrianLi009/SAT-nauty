# SAT + nauty: Orderly Generation of Small Kochen-Specker Sets

This repository contains the implementation and tools for exhaustively enumerating Kochen-Specker (KS) sets in dimension 3 using a novel SAT-based orderly generation framework that integrates recursive canonical labeling (RCL) with the graph isomorphism tool **nauty**.

## Overview

This work demonstrates the first exhaustive search for Kochen-Specker sets up to 33 rays containing the complete 25-ray state-independent contextuality (SI-C) set. We introduce a **SAT+nauty** framework that overcomes the exponential scaling issues of previous lexicographical approaches, enabling verification that the 33-ray Schütte set is the unique geometrically realizable KS set in this class.

### Key Contributions

1. **SAT+nauty Framework**: A novel integration of SAT solving with nauty's graph canonization via Recursive Canonical Labeling (RCL), maintaining fast canonicity checks on large graphs.

2. **Exhaustive Enumeration**: Complete classification of all KS sets up to 33 rays extending the complete 25-ray SI-C set (1,641 CPU hours).

3. **Verifiable Results**: All non-existence results are backed by independently verifiable DRAT proof certificates with domain-specific extensions for isomorph-rejection and geometric constraints.

## Repository Structure

```
├── scripts/               # Main workflow and automation
│   ├── automate_workflow.py    # Complete SAT solving workflow automation
│   ├── main.py                 # Main entry point for graph processing
│   └── parallel-solve.py       # Parallel solving utilities
│
├── graph_analysis/        # Graph creation and SIC analysis
│   ├── SI-C.py                 # Complete SIC graph creation
│   ├── SI-C-simple.py          # Minimal SIC graph creation
│   ├── orthogonality.py        # Orthogonality checking for graphs
│   ├── get_subgraphs.py        # Subgraph extraction utilities
│   └── minimal_conflicts.py    # Conflict analysis
│
├── verifiers/            # Proof verification tools
│   ├── drat_verifier.py        # DRAT proof verifier
│   ├── orthogonality_verifier.py  # Orthogonality constraint verifier
│   └── unified_verifier.py     # Unified verification
│
├── utils/                # Utility functions
│   ├── simple_graph6_to_canonical.py  # Graph6 to canonical form converter
│   ├── apply_mapping_to_vectors.py    # Vector mapping utilities
│   ├── add_vars_to_cnf.py      # CNF manipulation
│   ├── append.py               # Clause appending utilities
│   ├── to_var.py               # Variable conversion utilities
│   └── complete_basis.py       # Basis completion for vectors
│
├── tests/                # Test utilities
│   ├── test_010.py             # Testing utilities
│   ├── test_embed_simplified.py  # Embedding tests
│   └── sic13-coordinates.py    # SIC-13 coordinate definitions
│
├── cadical-rcl/          # Modified CaDiCaL solver with RCL integration
├── nauty2_8_8/           # nauty graph isomorphism tool
├── drat-trim/            # DRAT proof verification
├── gen_cubes/            # Cube generation utilities
└── gen_instance/         # CNF instance generation
```

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler (g++ or clang++)

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build solvers and tools
./setup.sh
```

The `setup.sh` script builds:
- **CaDiCaL-RCL** - Modified CaDiCaL with RCL propagator (`cadical-rcl/build/cadical`)

## Methodology

### Problem Definition

A **Kochen-Specker (KS) set** is a finite set of rays in Hilbert space ℂ³ that admits no 010-coloring: an assignment f: V → {0,1} such that:
- Adjacent (orthogonal) vertices are not both colored 1
- Each orthonormal basis (triangle) has exactly one vertex colored 1

The **state-independent contextuality (SI-C) set** is the minimal 13-ray Yu-Oh set that witnesses quantum contextuality independent of system state.

### SAT Encoding

Our encoding uses Boolean variables:
- `e_{i,j}`: vertices i and j are adjacent (rays are orthogonal)
- `t_{i,j,k}`: rays i, j, k form a mutually orthogonal triple

Key constraints enforced:
1. **Structural**: squarefree (no 4-cycles), minimum degree 3, every vertex in a triangle
2. **Non-010-colorability**: blocking clauses for all valid colorings
3. **Fixed subgraphs**: canonical base configuration (13-25 rays)

### Recursive Canonical Labeling (RCL)

Traditional lexicographical canonicity checks scale exponentially. RCL transforms nauty's efficient but non-hereditary canonical labeling into a hierarchical form:

1. Compute base labeling π = nauty(G)
2. Fix vertex v with largest label n
3. Recursively canonize G \ {v}

This ensures the **hereditary property**: if G is canonical, all prefixes G[k] are canonical, enabling efficient pruning during SAT search.

### Orthogonality Check

Domain-specific geometric propagator:
- **Fixed vectors**: Pre-calculated coordinates for base SI-C set
- **Derived vectors**: Computed on-the-fly via cross product when two orthogonal neighbors assigned
- **Conflict detection**: Parallel vectors or orthogonality violations trigger minimal blocking clauses
- **Exact arithmetic**: All computations use ℤ³ or algebraic fields to avoid floating-point errors

## Usage

This repository is designed to reproduce the main result from the paper: **exhaustive enumeration of Kochen-Specker sets extending the complete 25-ray SI-C set up to order 33.**

### Quick Start: One-Command Workflow

The simplest way to reproduce results:

```bash
# Build tools (one-time setup)
./setup.sh

# Run complete workflow for order 28
python3 run_sic25_search.py --order 28

# Run complete workflow for order 33 (full paper experiment)
python3 run_sic25_search.py --order 33
```

This single script:
1. Generates base CNF constraints for the target order
2. Adds the 25-ray SI-C unit clauses from `sic-25.vars`
3. Runs CaDiCaL-RCL with the 25-ray SI-C vector coordinates from `sic-25-vectors.txt`
4. Saves results to `results/` directory

**Options:**
- `--order N`: Target graph order (default: 28)
- `--output-dir DIR`: Output directory (default: results)
- `--complex`: Use complex arithmetic for complex-valued KS sets
- `--skip-generation`: Skip CNF generation if file exists

### Manual Workflow (Step-by-Step)

If you want to run each step manually:

1. **Build the tools** (one-time setup)
2. **Generate CNF constraints** for the target order
3. **Run the SAT+nauty solver** with the 25-ray base configuration
4. **Verify results** (optional but recommended)

#### Step 1: Build Tools

```bash
# Build CaDiCaL-RCL solver
./setup.sh
```

This compiles:
- `cadical-rcl/build/cadical` - Modified CaDiCaL with RCL integration

#### Step 2: Generate CNF Instance

```bash
# Generate constraints for order N (e.g., N=28)
# Parameters: <order> <block_ratio> <lex_option>
cd gen_instance
python3 generate.py 28 0 no-lex
cd ..

# This creates: gen_instance/constraints_28_0_no_lex
```

**Parameters:**
- `order`: Target graph size (28-33 for paper results)
- `block_ratio`: Ratio for 010-coloring blocking (use 0 for standard)
- `lex_option`: Use `no-lex` (RCL handles canonicity, not lex-based)

#### Step 2.5: Add 25-ray SI-C Constraints

```bash
# Add the 25-ray SI-C unit clauses to the base CNF
python3 utils/add_vars_to_cnf.py \
    gen_instance/constraints_28_0_no_lex \
    sic-25.vars \
    constraints_28_0_no_lex_sic25.cnf
```

This adds the fixed 25-ray SI-C configuration as unit clauses.

#### Step 3: Run SAT+nauty Solver

```bash
# Run solver on the CNF with SI-C constraints
./solve-verify.sh \
    --order 28 \
    --partition 25 \
    --vectors-file sic-25-vectors.txt \
    constraints_28_0_no_lex_sic25.cnf
```

**Key Parameters:**
- `--order N`: Graph order (must match CNF generation)
- `--partition K`: Starting subgraph size (25 for complete SI-C set)
- `--vectors-file`: Vector coordinates for the 25-ray SI-C (`sic-25-vectors.txt`)
- `--ortho`: Enable orthogonality propagator (optional, enabled by default)
- `--complex`: Use complex arithmetic (for complex-valued KS sets)

**Output:**
- Results are logged to `<cnf_file>.log`
- DRAT proof trace (if proof logging enabled in solver build)
- Found KS sets (if SAT) or UNSAT confirmation

#### Step 4: Verify Results (Optional)

For UNSAT results with proof traces:

```bash
# Verify extended DRAT proof
cd verifiers
python3 unified_verifier.py \
    --cnf constraints_28_0_no_lex \
    --proof constraints_28_0_no_lex.drat
```

### Example: Complete Search for 33-ray KS Sets

Using the simple one-command workflow:

```bash
# 1. Build tools (one-time)
./setup.sh

# 2. Run complete search
python3 run_sic25_search.py --order 33
```

Or manually:

```bash
# 1. Build tools
./setup.sh

# 2. Generate CNF for order 33
cd gen_instance
python3 generate.py 33 0 no-lex
cd ..

# 3. Add SI-C constraints
python3 utils/add_vars_to_cnf.py \
    gen_instance/constraints_33_0_no_lex \
    sic-25.vars \
    constraints_33_0_no_lex_sic25.cnf

# 4. Run solver
./solve-verify.sh \
    --order 33 \
    --partition 25 \
    --vectors-file sic-25-vectors.txt \
    constraints_33_0_no_lex_sic25.cnf
```

**Note:** Order 33 search takes significant computational resources (~1,641 CPU hours as reported in paper). For testing, try smaller orders (28-30).

## Key Results

### Performance Comparison: RCL vs Lexicographical Canonicity

Traditional lexicographical approaches become intractable on canonical graphs at order N ≥ 30.

| Method | Canonical Check Time | Order 33 Feasibility |
|--------|---------------------|---------------------|
| Lex-based (SAT+CAS, SMS) | Exponential (~timeout) | ❌ Intractable |
| **RCL (SAT+nauty)** | **0.008s constant** | **✓ Tractable** |

**Benchmark on 400 canonical graphs (N=30-33):**
- **SAT+nauty (RCL)**: 100% solved, avg 0.004s, PAR-2 = 0.0038
- **Optimized Lex**: 80.5% solved, avg 4.67s, PAR-2 = 27.16

### Enumeration Results

**Complete 25-ray SI-C Extension to Order 33:**
- Computational effort: 1,641 CPU hours (~68 days, parallelized to 2 days wall-clock)
- Combinatorial candidates found: 44 non-isomorphic graphs
- **Geometrically realizable: 1** (Schütte 33-ray set)
- Verified uniqueness of Schütte set containing complete 25-ray SI-C core
- Proof certificate: ~13 TiB extended DRAT format, independently verified

**Key Finding:** The 33-ray Kochen-Specker set discovered by Schütte is the **unique** geometrically realizable KS set extending the complete 25-ray state-independent contextuality set.

## Proof Verification

Our extended DRAT format includes:
- **t-clauses**: Canonicity axioms with permutation witnesses
- **o-clauses**: Orthogonality violation witnesses with minimal edge sets
- Standard RUP (Reverse Unit Propagation) for learned clauses

Verification guarantees:
1. No canonical representative was blocked (soundness of isomorph-rejection)
2. No geometrically valid graph was blocked (soundness of geometric pruning)
3. Complete coverage of search space

## Dependencies

- **CaDiCaL**: SAT solver (modified with RCL propagator)
- **nauty**: Graph isomorphism and canonical labeling
- **drat-trim**: DRAT proof verification
- **NetworkX**: Graph manipulation in Python
- **NumPy**: Numerical computations
- **PySAT**: Python SAT solver interface

## License

See individual subdirectories for component licenses:
- `cadical-rcl/`: MIT License
- `nauty2_8_8/`: Apache 2.0 License
- `drat-trim/`: MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the authors.

## Acknowledgments

This work demonstrates the power of combining domain-specific propagators with state-of-the-art graph isomorphism tools. The SAT+nauty framework is general and applicable to other combinatorial generation problems requiring efficient symmetry handling.


