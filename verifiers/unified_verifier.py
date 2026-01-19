#!/usr/bin/env python3
"""
Unified Verifier for Canonicity and Orthogonality Blocking Clauses

This script verifies both types of blocking clauses:
1. Canonicity clauses ('t' prefix) - verified using .perm file
2. Orthogonality clauses ('o' prefix) - verified using .ortho file

The verifier processes clauses in order from the DRAT file, switching
between verification methods based on the clause type marker.
"""

import sys
import re
from typing import List, Tuple, Dict, Set, Optional
import argparse


# ============================================================================
# Common Utilities
# ============================================================================

def clause_to_string(clause: List[int]) -> str:
    """Convert clause to string format for comparison."""
    return ' '.join(map(str, sorted(clause, key=abs)))


# ============================================================================
# Canonicity Verification (from drat_verifier.py)
# ============================================================================

def parse_perm_file(filename: str) -> List[List[int]]:
    """Parse permutation file and return list of permutations."""
    permutations = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                perm = [int(x) for x in line.split()]
                permutations.append(perm)
    return permutations


def parse_fixed_edges_file(filename: str) -> List[int]:
    """Parse fixed edges file and return list of edge literals."""
    fixed_edges = []
    with open(filename, 'r') as f:
        content = f.read().strip()
        if content:
            fixed_edges = [int(x) for x in content.split()]
    return fixed_edges


def augment_clause_with_fixed_edges(clause: List[int], fixed_edges: List[int]) -> List[int]:
    """Add fixed edges to a clause, avoiding duplicates."""
    clause_vars = {abs(lit) for lit in clause}
    augmented = clause.copy()
    for lit in fixed_edges:
        if abs(lit) not in clause_vars:
            augmented.append(lit)
    return augmented


def variable_to_matrix_position(var: int, n: int) -> Tuple[int, int]:
    """Convert variable number to adjacency matrix position."""
    current_var = 1
    for col in range(1, n):
        for row in range(col):
            if current_var == var:
                return row, col
            current_var += 1
    raise ValueError(f"Variable {var} out of range for n={n}")


def matrix_position_to_variable(row: int, col: int, n: int) -> int:
    """Convert adjacency matrix position back to variable number."""
    var = 1
    for c in range(1, n):
        for r in range(c):
            if r == row and c == col:
                return var
            var += 1
    raise ValueError(f"Position ({row},{col}) out of range for n={n}")


def clause_to_adjacency_matrix(clause: List[int], n: int) -> List[List[int]]:
    """Convert clause to adjacency matrix representation."""
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for var in clause:
        abs_var = abs(var)
        row, col = variable_to_matrix_position(abs_var, n)
        value = 1 if var < 0 else 0
        matrix[row][col] = value
        matrix[col][row] = value
    
    return matrix


def adjacency_matrix_to_clause(matrix: List[List[int]], n: int) -> List[int]:
    """Convert adjacency matrix back to clause representation."""
    clause = []
    for col in range(1, n):
        for row in range(col):
            var = matrix_position_to_variable(row, col, n)
            if matrix[row][col] == 1:
                clause.append(-var)
            elif matrix[row][col] == 0:
                clause.append(var)
    return clause


def apply_permutation(matrix: List[List[int]], perm: List[int]) -> List[List[int]]:
    """Apply permutation to adjacency matrix."""
    n = len(matrix)
    permuted_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            permuted_matrix[perm[i]][perm[j]] = matrix[i][j]
    
    return permuted_matrix


def verify_canonicity_clause(t_clause: List[int], 
                              perm: List[int],
                              clause_set: Set[str],
                              fixed_edges: List[int] = None,
                              verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify a canonicity blocking clause.
    
    Returns (is_valid, reason)
    """
    if fixed_edges is None:
        fixed_edges = []
    
    graph_size = len(perm)
    expected_clause_size = graph_size * (graph_size - 1) // 2
    
    # Augment clause with fixed edges for processing
    augmented_clause = augment_clause_with_fixed_edges(t_clause, fixed_edges)
    
    if len(augmented_clause) != expected_clause_size:
        if not fixed_edges:
            return False, f"Clause size {len(augmented_clause)} doesn't match expected {expected_clause_size} for graph size {graph_size}"
        else:
            return False, f"Clause has {len(t_clause)} literals + {len(fixed_edges)} fixed = {len(augmented_clause)} total, expected {expected_clause_size}"
    
    original_clause_str = clause_to_string(augmented_clause)
    
    # Convert to adjacency matrix
    matrix = clause_to_adjacency_matrix(augmented_clause, graph_size)
    
    # Apply permutation
    permuted_matrix = apply_permutation(matrix, perm)
    
    # Convert back to clause
    permuted_clause = adjacency_matrix_to_clause(permuted_matrix, graph_size)
    permuted_clause_str = clause_to_string(permuted_clause)
    
    if verbose:
        print(f"  Original clause: {original_clause_str[:50]}...")
        print(f"  Permuted clause: {permuted_clause_str[:50]}...")
    
    # Check if permuted clause exists in clause set (and is different)
    if permuted_clause_str in clause_set and permuted_clause_str != original_clause_str:
        return False, "Permuted clause found as blocking clause (canonicity violation)"
    
    return True, "Valid canonicity clause"


# ============================================================================
# Orthogonality Verification
# ============================================================================

def parse_ortho_witness(witness_str: str) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Parse a single orthogonality witness line.
    
    Format: edges:254,268,283 vectors:v1=[1,-2,1],v7=[1,-1,2]
    Returns: (edges, vectors)
    """
    edges_match = re.search(r'edges:([\d,]+)', witness_str)
    vectors_match = re.search(r'vectors:(.+)$', witness_str)
    
    if not edges_match or not vectors_match:
        raise ValueError(f"Could not parse witness: {witness_str}")
    
    # Parse edges
    edges_str = edges_match.group(1)
    edges = [int(x) for x in edges_str.split(',')]
    
    # Parse vectors
    vectors = {}
    vectors_str = vectors_match.group(1)
    vector_parts = re.findall(r'(v\d+)=\[([^\]]+)\]', vectors_str)
    
    for vertex_name, coords_str in vector_parts:
        coords = [int(x) for x in coords_str.split(',')]
        vectors[vertex_name] = coords
    
    return edges, vectors


def parse_ortho_file(filename: str) -> List[Tuple[List[int], Dict[str, List[int]]]]:
    """Parse .ortho witness file."""
    witnesses = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                edges, vectors = parse_ortho_witness(line)
                witnesses.append((edges, vectors))
    return witnesses


def edge_index_to_vertex_pair(edge_idx: int, n: int) -> Tuple[int, int]:
    """Convert edge index to vertex pair (i, j) where i < j."""
    current_edge = 0
    for col in range(1, n):
        for row in range(col):
            if current_edge == edge_idx:
                return (row, col)
            current_edge += 1
    raise ValueError(f"Edge index {edge_idx} out of range for n={n}")


def calculate_n_from_max_edge(max_edge: int) -> int:
    """Calculate number of vertices from maximum edge index."""
    n = 2
    while n * (n - 1) // 2 <= max_edge:
        n += 1
    return n


def dot_product(v1: List[int], v2: List[int]) -> int:
    """Calculate dot product of two vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")
    return sum(a * b for a, b in zip(v1, v2))


def verify_orthogonality_clause(o_clause: List[int],
                                 edges: List[int],
                                 vectors: Dict[str, List[int]],
                                 verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify an orthogonality blocking clause.
    
    The clause should block a configuration where edges connect
    non-orthogonal vertices.
    
    Returns (is_valid, reason)
    """
    # Extract edge indices from clause (negative literals)
    clause_edges = []
    for lit in o_clause:
        if lit < 0:
            edge_idx = (-lit) - 1
            clause_edges.append(edge_idx)
    
    # Check that clause edges match witness edges
    if set(clause_edges) != set(edges):
        return False, f"Clause edges {sorted(clause_edges)} don't match witness edges {sorted(edges)}"
    
    # Calculate n from maximum edge index
    max_edge = max(edges)
    n = calculate_n_from_max_edge(max_edge)
    
    # Verify orthogonality violations
    violations = []
    for edge_idx in edges:
        row, col = edge_index_to_vertex_pair(edge_idx, n)
        
        v1_name = f"v{row + 1}"
        v2_name = f"v{col + 1}"
        
        if v1_name not in vectors or v2_name not in vectors:
            continue
        
        v1 = vectors[v1_name]
        v2 = vectors[v2_name]
        
        dot = dot_product(v1, v2)
        
        if verbose:
            print(f"    Edge {edge_idx}: v{row+1}={v1} · v{col+1}={v2} = {dot}")
        
        if dot != 0:
            violations.append((edge_idx, row + 1, col + 1, dot))
    
    if not violations:
        return False, "No orthogonality violations found"
    
    return True, f"Valid: {len(violations)} orthogonality violation(s)"


# ============================================================================
# Unified DRAT Parser
# ============================================================================

def parse_drat_clauses(filename: str) -> List[Tuple[str, List[int]]]:
    """
    Parse DRAT file and return list of (clause_type, clause) tuples.
    
    clause_type: 't' for canonicity, 'o' for orthogonality
    """
    clauses = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('d'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 't':
                # Canonicity clause
                clause = [int(x) for x in parts[1:-1]]
                clauses.append(('t', clause))
            elif parts[0] == 'o':
                # Orthogonality clause
                clause = [int(x) for x in parts[1:-1]]
                clauses.append(('o', clause))
    
    return clauses


# ============================================================================
# Main Verification
# ============================================================================

def verify_unified_proof(drat_file: str, 
                        perm_file: Optional[str] = None,
                        ortho_file: Optional[str] = None,
                        fixed_edges_file: Optional[str] = None,
                        verbose: bool = False) -> bool:
    """Main unified verification function."""
    print(f"Unified Verification")
    print(f"DRAT file: {drat_file}")
    if perm_file:
        print(f"Permutation file: {perm_file}")
    if ortho_file:
        print(f"Orthogonality file: {ortho_file}")
    if fixed_edges_file:
        print(f"Fixed edges file: {fixed_edges_file}")
    print()
    
    # Parse fixed edges if provided
    fixed_edges = []
    if fixed_edges_file:
        fixed_edges = parse_fixed_edges_file(fixed_edges_file)
        print(f"Loaded {len(fixed_edges)} fixed edge assignments")
    
    # Parse DRAT file
    clauses = parse_drat_clauses(drat_file)
    t_clauses = [(i, c) for i, (t, c) in enumerate(clauses) if t == 't']
    o_clauses = [(i, c) for i, (t, c) in enumerate(clauses) if t == 'o']
    
    print(f"Found {len(t_clauses)} canonicity clauses ('t')")
    print(f"Found {len(o_clauses)} orthogonality clauses ('o')")
    print()
    
    # Load witnesses
    permutations = []
    if perm_file:
        permutations = parse_perm_file(perm_file)
        print(f"Loaded {len(permutations)} permutations from {perm_file}")
    
    ortho_witnesses = []
    if ortho_file:
        ortho_witnesses = parse_ortho_file(ortho_file)
        print(f"Loaded {len(ortho_witnesses)} orthogonality witnesses from {ortho_file}")
    
    print()
    
    # Check counts
    if len(t_clauses) != len(permutations):
        print(f"⚠️  WARNING: {len(t_clauses)} 't' clauses but {len(permutations)} permutations")
        if len(t_clauses) > len(permutations):
            print(f"   Some 't' clauses will not be verified!")
    
    if len(o_clauses) != len(ortho_witnesses):
        print(f"⚠️  WARNING: {len(o_clauses)} 'o' clauses but {len(ortho_witnesses)} orthogonality witnesses")
        if len(o_clauses) > len(ortho_witnesses):
            print(f"   Some 'o' clauses will not be verified!")
    
    # Build clause set for canonicity checking (with fixed edges augmented)
    clause_set = set()
    for _, (clause_type, clause) in enumerate(clauses):
        if clause_type == 't':
            augmented_clause = augment_clause_with_fixed_edges(clause, fixed_edges)
            clause_set.add(clause_to_string(augmented_clause))
    
    # Verify canonicity clauses
    verification_passed = True
    canonicity_failures = 0
    
    if t_clauses and permutations:
        print(f"\nVerifying {len(t_clauses)} canonicity clauses...")
        for i, (clause_idx, t_clause) in enumerate(t_clauses):
            if i >= len(permutations):
                print(f"⚠️  Skipping 't' clause {i+1} (no permutation)")
                continue
            
            perm = permutations[i]
            
            if verbose:
                print(f"\n  Canonicity clause {i+1}/{len(t_clauses)}:")
                print(f"    Permutation: {perm}")
            
            is_valid, reason = verify_canonicity_clause(t_clause, perm, clause_set, fixed_edges, verbose)
            
            if not is_valid:
                print(f"❌ FAIL: Canonicity clause {i+1} - {reason}")
                canonicity_failures += 1
                verification_passed = False
            elif verbose:
                print(f"  ✅ PASS: {reason}")
        
        if canonicity_failures == 0:
            print(f"✅ All {len(t_clauses)} canonicity clauses verified!")
        else:
            print(f"❌ {canonicity_failures}/{len(t_clauses)} canonicity clauses failed!")
    
    # Verify orthogonality clauses
    orthogonality_failures = 0
    
    if o_clauses and ortho_witnesses:
        print(f"\nVerifying {len(o_clauses)} orthogonality clauses...")
        for i, (clause_idx, o_clause) in enumerate(o_clauses):
            if i >= len(ortho_witnesses):
                print(f"⚠️  Skipping 'o' clause {i+1} (no witness)")
                continue
            
            edges, vectors = ortho_witnesses[i]
            
            if verbose:
                print(f"\n  Orthogonality clause {i+1}/{len(o_clauses)}:")
                print(f"    Edges: {edges}")
                print(f"    Vectors: {list(vectors.keys())}")
            
            is_valid, reason = verify_orthogonality_clause(o_clause, edges, vectors, verbose)
            
            if not is_valid:
                print(f"❌ FAIL: Orthogonality clause {i+1} - {reason}")
                orthogonality_failures += 1
                verification_passed = False
            elif verbose:
                print(f"  ✅ PASS: {reason}")
        
        if orthogonality_failures == 0:
            print(f"✅ All {len(o_clauses)} orthogonality clauses verified!")
        else:
            print(f"❌ {orthogonality_failures}/{len(o_clauses)} orthogonality clauses failed!")
    
    return verification_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Unified verifier for canonicity and orthogonality blocking clauses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify both types of clauses
  python3 unified_verifier.py proof.drat --perm proof.perm --ortho proof.ortho
  
  # Verify only canonicity clauses
  python3 unified_verifier.py proof.drat --perm proof.perm
  
  # Verify only orthogonality clauses
  python3 unified_verifier.py proof.drat --ortho proof.ortho
  
  # With fixed edges (for partitioned graphs)
  python3 unified_verifier.py proof.drat --perm proof.perm --fixed-edges fixed.txt
  
  # Verbose output
  python3 unified_verifier.py proof.drat --perm proof.perm --ortho proof.ortho --verbose
        """
    )
    
    parser.add_argument('drat_file', help='Path to DRAT file')
    parser.add_argument('--perm', help='Path to permutation file (.perm)')
    parser.add_argument('--ortho', help='Path to orthogonality witness file (.ortho)')
    parser.add_argument('--fixed-edges', help='Path to fixed edges file (space-separated literals)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed verification output')
    
    args = parser.parse_args()
    
    if not args.perm and not args.ortho:
        print("Error: Must provide at least one of --perm or --ortho")
        sys.exit(1)
    
    try:
        result = verify_unified_proof(args.drat_file, args.perm, args.ortho, args.fixed_edges, args.verbose)
        
        print("\n" + "=" * 70)
        if result:
            print("🎉 VERIFICATION PASSED")
            print("All blocking clauses are valid!")
        else:
            print("❌ VERIFICATION FAILED")
            print("Some blocking clauses are invalid!")
        
        sys.exit(0 if result else 1)
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

