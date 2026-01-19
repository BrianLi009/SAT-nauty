#!/usr/bin/env python3
"""
Orthogonality Constraint Verifier

This script verifies orthogonality blocking clauses by checking that the witness
vectors indeed violate orthogonality constraints.

For each 'o' clause in the DRAT file:
1. Extract the blocking clause (edges that cannot all be present)
2. Read the corresponding witness from .ortho file
3. Verify that with the given vertex coordinates, at least one edge pair
   would violate orthogonality (dot product ≠ 0)
"""

import sys
import re
from typing import List, Tuple, Dict, Set
import argparse


def parse_ortho_clauses(drat_file: str) -> List[List[int]]:
    """Parse 'o' clauses from DRAT file."""
    ortho_clauses = []
    
    with open(drat_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('o '):
                parts = line.split()
                # Extract clause (remove 'o' and '0')
                clause = [int(x) for x in parts[1:-1]]
                ortho_clauses.append(clause)
    
    return ortho_clauses


def parse_ortho_witness_file(ortho_file: str) -> List[Tuple[List[int], Dict[str, List[int]]]]:
    """
    Parse .ortho witness file.
    
    Returns list of (edges, vectors) tuples where:
    - edges: list of edge indices
    - vectors: dict mapping vertex names to coordinate lists
    """
    witnesses = []
    
    with open(ortho_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: edges:254,268,283 vectors:v1=[1,-2,1],v7=[1,-1,2]
            edges_match = re.search(r'edges:([\d,]+)', line)
            vectors_match = re.search(r'vectors:(.+)$', line)
            
            if not edges_match or not vectors_match:
                print(f"Warning: Could not parse witness line: {line}")
                continue
            
            # Parse edges
            edges_str = edges_match.group(1)
            edges = [int(x) for x in edges_str.split(',')]
            
            # Parse vectors
            vectors = {}
            vectors_str = vectors_match.group(1)
            # Split by comma, but be careful with coordinates like [1,-2,1]
            vector_parts = re.findall(r'(v\d+)=\[([^\]]+)\]', vectors_str)
            
            for vertex_name, coords_str in vector_parts:
                coords = [int(x) for x in coords_str.split(',')]
                vectors[vertex_name] = coords
            
            witnesses.append((edges, vectors))
    
    return witnesses


def edge_index_to_vertex_pair(edge_idx: int, n: int) -> Tuple[int, int]:
    """
    Convert edge index to vertex pair (i, j) where i < j.
    
    Edges are numbered column by column in upper triangle.
    For n vertices: edge 0 = (0,1), edge 1 = (0,2), ..., edge n-2 = (0,n-1),
                    edge n-1 = (1,2), edge n = (1,3), ...
    """
    current_edge = 0
    for col in range(1, n):
        for row in range(col):
            if current_edge == edge_idx:
                return (row, col)
            current_edge += 1
    
    raise ValueError(f"Edge index {edge_idx} out of range for n={n}")


def calculate_n_from_max_edge(max_edge: int) -> int:
    """
    Calculate number of vertices from maximum edge index.
    
    For n vertices, we have n*(n-1)/2 edges (0 to n*(n-1)/2 - 1).
    So we need to find n such that n*(n-1)/2 > max_edge.
    """
    n = 2
    while n * (n - 1) // 2 <= max_edge:
        n += 1
    return n


def dot_product(v1: List[int], v2: List[int]) -> int:
    """Calculate dot product of two vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")
    return sum(a * b for a, b in zip(v1, v2))


def verify_orthogonality_clause(clause: List[int], 
                                 edges: List[int], 
                                 vectors: Dict[str, List[int]],
                                 verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify that an orthogonality blocking clause is correct.
    
    The clause should block a configuration where:
    - All edges in the clause are present (negative literals)
    - With the given vector assignments, at least one edge connects
      non-orthogonal vertices
    
    Returns (is_valid, reason)
    """
    # Convert clause literals to edge indices
    clause_edges = []
    for lit in clause:
        if lit < 0:
            # Negative literal means edge should be absent
            # Variable -lit corresponds to edge index (-lit - 1)
            edge_idx = (-lit) - 1
            clause_edges.append(edge_idx)
        else:
            # Positive literals in blocking clauses are unusual
            # They mean "edge is absent" in the blocked configuration
            if verbose:
                print(f"  Warning: Positive literal {lit} in blocking clause (edge should be absent)")
    
    # Check that clause edges match witness edges
    clause_edges_set = set(clause_edges)
    witness_edges_set = set(edges)
    
    if clause_edges_set != witness_edges_set:
        return False, f"Clause edges {sorted(clause_edges)} don't match witness edges {sorted(edges)}"
    
    if verbose:
        print(f"  Clause edges: {sorted(clause_edges)}")
        print(f"  Witness edges: {sorted(edges)}")
    
    # Calculate n (number of vertices) from maximum edge index
    max_edge = max(edges)
    n = calculate_n_from_max_edge(max_edge)
    
    if verbose:
        print(f"  Calculated n={n} vertices from max edge {max_edge}")
    
    # For each edge, verify that it violates orthogonality
    violations = []
    for edge_idx in edges:
        row, col = edge_index_to_vertex_pair(edge_idx, n)
        
        # Get vector names (vertices are 0-indexed, but witness uses 1-indexed)
        v1_name = f"v{row + 1}"
        v2_name = f"v{col + 1}"
        
        if v1_name not in vectors or v2_name not in vectors:
            if verbose:
                print(f"  Warning: Edge {edge_idx} = ({row},{col}) missing vectors in witness")
            continue
        
        v1 = vectors[v1_name]
        v2 = vectors[v2_name]
        
        dot = dot_product(v1, v2)
        
        if verbose:
            print(f"  Edge {edge_idx}: v{row+1}={v1} · v{col+1}={v2} = {dot}")
        
        if dot != 0:
            violations.append((edge_idx, row + 1, col + 1, dot))
    
    if not violations:
        return False, f"No orthogonality violations found! All edges connect orthogonal vertices."
    
    if verbose:
        print(f"  ✓ Found {len(violations)} orthogonality violation(s):")
        for edge_idx, v1_idx, v2_idx, dot in violations:
            print(f"    Edge {edge_idx} (v{v1_idx}, v{v2_idx}): dot product = {dot} ≠ 0")
    
    return True, f"Valid: {len(violations)} orthogonality violations found"


def verify_orthogonality_proof(drat_file: str, ortho_file: str, verbose: bool = False) -> bool:
    """Main verification function."""
    print(f"Verifying orthogonality constraints")
    print(f"DRAT file: {drat_file}")
    print(f"Orthogonality witness file: {ortho_file}")
    print()
    
    # Parse DRAT file for 'o' clauses
    ortho_clauses = parse_ortho_clauses(drat_file)
    print(f"Found {len(ortho_clauses)} orthogonality clauses in DRAT file")
    
    # Parse witness file
    witnesses = parse_ortho_witness_file(ortho_file)
    print(f"Found {len(witnesses)} witnesses in .ortho file")
    print()
    
    if len(ortho_clauses) != len(witnesses):
        print(f"❌ ERROR: Mismatch between number of 'o' clauses ({len(ortho_clauses)}) "
              f"and witnesses ({len(witnesses)})")
        return False
    
    if len(ortho_clauses) == 0:
        print("ℹ️  No orthogonality clauses to verify")
        return True
    
    # Verify each clause
    verification_passed = True
    for i, (clause, (edges, vectors)) in enumerate(zip(ortho_clauses, witnesses)):
        if verbose:
            print(f"Verifying orthogonality clause {i+1}/{len(ortho_clauses)}:")
            print(f"  Clause: {clause}")
        
        is_valid, reason = verify_orthogonality_clause(clause, edges, vectors, verbose)
        
        if not is_valid:
            print(f"❌ FAIL: Orthogonality clause {i+1} verification failed!")
            print(f"   Reason: {reason}")
            verification_passed = False
        else:
            if verbose:
                print(f"✅ PASS: Orthogonality clause {i+1} verified - {reason}")
        
        if verbose:
            print()
    
    return verification_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Verify orthogonality blocking clauses in DRAT proofs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 orthogonality_verifier.py proof.drat proof.ortho
  python3 orthogonality_verifier.py proof.drat proof.ortho --verbose
        """
    )
    
    parser.add_argument('drat_file', help='Path to DRAT file')
    parser.add_argument('ortho_file', help='Path to orthogonality witness file (.ortho)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed verification output')
    
    args = parser.parse_args()
    
    try:
        result = verify_orthogonality_proof(args.drat_file, args.ortho_file, args.verbose)
        
        print("=" * 50)
        if result:
            print("🎉 ORTHOGONALITY VERIFICATION PASSED")
            print("All orthogonality blocking clauses are valid!")
        else:
            print("❌ ORTHOGONALITY VERIFICATION FAILED")
            print("Some orthogonality clauses are invalid!")
        
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

