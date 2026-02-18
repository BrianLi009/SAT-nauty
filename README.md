# SAT + nauty: Exhaustive Enumeration of Kochen-Specker Sets

This repository searches for Kochen-Specker (KS) sets in dimension 3 extending the complete 25-ray state-independent contextuality (SI-C) set, using a SAT-based framework integrated with nauty's recursive canonical labeling (RCL).

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build CaDiCaL-RCL solver
./setup.sh
```

## Usage

```bash
# Run search for order 28
python3 run_sic25_search.py --order 28

# Run search for order 33
python3 run_sic25_search.py --order 33

# Run search for order 30 and generate a proof
python3 run_sic25_search.py --order 30 --proof
```

The script:
1. Generates base CNF constraints for the target order
2. Adds the 25-ray SI-C unit clauses from `sic-25.vars`
3. Runs CaDiCaL-RCL with the SI-C vector coordinates from `sic-25-vectors.txt`
4. Saves results to `results/`

**Options:**
- `--order N`: Target graph order (default: 28)
- `--output-dir DIR`: Output directory (default: results)
- `--partition N`: Starting partition size (default: 25)
- `--complex`: Use complex arithmetic
- `--proof`: Generate a DRAT proof file
- `--binary`: Use binary DRAT format (default: text)
- `--skip-generation`: Skip CNF generation if file exists

## Proof Generation and Verification

The full verification pipeline has three steps: solve with proof generation, verify the DRAT proof, and verify the domain-specific clauses.

### Step 1: Solve with proof generation

```bash
python3 run_sic25_search.py --order 30 --proof
```

This generates the CNF, runs CaDiCaL-RCL, and produces:
- `results/order_constraints_30_0_no_lex_sic25.cnf` — the CNF instance
- `results/order_constraints_30_0_no_lex_sic25.drat` — the DRAT proof
- `results/order_constraints_30_0_no_lex_sic25.perm` — canonicity permutation witness file
- `results/order_constraints_30_0_no_lex_sic25.ortho` — orthogonality witness file

The DRAT proof contains domain-specific `t` (canonicity) and `o` (orthogonality) prefixed clauses from the external propagator.

### Step 2: Verify DRAT proof with drat-trim

```bash
./drat-trim/drat-trim results/order_constraints_30_0_no_lex_sic25.cnf results/order_constraints_30_0_no_lex_sic25.drat -f
```

The included drat-trim is modified to accept `t` and `o` clauses as trusted axioms, skipping their RUP/RAT verification. This step confirms that the standard resolution steps in the proof are correct.

### Step 3: Verify domain-specific clauses with unified verifier

```bash
python3 verifiers/unified_verifier.py results/order_constraints_30_0_no_lex_sic25.drat --perm results/order_constraints_30_0_no_lex_sic25.perm --fixed-edges sic-25.vars --ortho results/order_constraints_30_0_no_lex_sic25.ortho --verbose
```

This independently verifies both types of domain-specific clauses:
- **`t` clauses (canonicity)**: Checks that each symmetry-breaking clause correctly identifies a non-canonical subgraph by verifying the canonical labeling via nauty, using the permutation witnesses from the `.perm` file and the fixed edges from `sic-25.vars`.
- **`o` clauses (orthogonality)**: Checks that each orthogonality blocking clause is justified by verifying that the witness vectors yield a non-zero dot product, confirming a genuine orthogonality violation.

Together, steps 2 and 3 provide a complete, independently checkable proof of unsatisfiability.

## License

See individual subdirectories for component licenses:
- `cadical-rcl/`: MIT License
- `nauty2_8_8/`: Apache 2.0 License
- `drat-trim/`: MIT License
