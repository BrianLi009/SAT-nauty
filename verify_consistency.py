#!/usr/bin/env python3
"""
Verify consistency between sic-25-vectors.txt and sic-25.vars
Check that edges correspond to orthogonal vectors.
"""

def dot_product(v1, v2):
    """Calculate dot product of two 3D vectors."""
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def parse_vectors(filename):
    """Parse vector coordinates from file."""
    vectors = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                coords = list(map(int, line.split()))
                if len(coords) == 3:
                    vectors.append(tuple(coords))
    return vectors

def parse_vars(filename):
    """Parse edge variables from sic-25.vars file."""
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    literals = list(map(int, content.split()))
    
    # Create adjacency matrix
    n = 25
    edges = []
    var_num = 1
    
    for j in range(n):
        for i in range(j):
            lit = None
            for l in literals:
                if abs(l) == var_num:
                    lit = l
                    break
            
            if lit and lit > 0:  # Positive literal means edge exists
                edges.append((i, j))
            
            var_num += 1
    
    return edges

def main():
    print("="*80)
    print("CONSISTENCY CHECK: sic-25-vectors.txt vs sic-25.vars")
    print("="*80)
    print()
    
    # Parse files
    vectors = parse_vectors('sic-25-vectors.txt')
    edges = parse_vars('sic-25.vars')
    
    print(f"Loaded {len(vectors)} vectors")
    print(f"Loaded {len(edges)} edges from sic-25.vars")
    print()
    
    if len(vectors) != 25:
        print(f"❌ ERROR: Expected 25 vectors, got {len(vectors)}")
        return 1
    
    # Check consistency
    errors = []
    correct = []
    
    for i, j in edges:
        v1 = vectors[i]
        v2 = vectors[j]
        dot = dot_product(v1, v2)
        
        if dot == 0:
            correct.append((i, j))
        else:
            errors.append((i, j, v1, v2, dot))
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    
    if len(errors) == 0:
        print(f"✓ ALL {len(edges)} EDGES ARE CONSISTENT!")
        print(f"  Every edge connects orthogonal vectors (dot product = 0)")
        print()
        print("Sample checks:")
        for idx, (i, j) in enumerate(correct[:5]):
            v1 = vectors[i]
            v2 = vectors[j]
            print(f"  Edge ({i:2d}, {j:2d}): v{i+1}={v1} ⊥ v{j+1}={v2} ✓")
        if len(correct) > 5:
            print(f"  ... and {len(correct)-5} more")
        
        return 0
    else:
        print(f"❌ FOUND {len(errors)} INCONSISTENT EDGES!")
        print()
        for i, j, v1, v2, dot in errors[:10]:
            print(f"  Edge ({i:2d}, {j:2d}): v{i+1}={v1} · v{j+1}={v2} = {dot} (NOT orthogonal!)")
        
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more errors")
        
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())

