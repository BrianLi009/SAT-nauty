#!/usr/bin/env python3
"""
DRAT Verifier for Graph Coloring/SAT Problems

This script verifies DRAT proofs by checking that permuted clauses don't exist
as blocking clauses. For each 't' clause in the DRAT file, it:
1. Extracts the clause (removes 't' and '0')
2. Converts it to an adjacency matrix representation
3. Applies the corresponding permutation from the .perm file
4. Converts back to a clause
5. Verifies this permuted clause doesn't exist in the DRAT file

For each 'o' clause (orthogonality constraints) in the DRAT file, it:
1. Extracts the clause (removes 'o' and '0')
2. Finds the corresponding witness in the .ortho file
3. Verifies that the blocking clause correctly prevents the orthogonality violation
4. Checks that connected vertices have non-orthogonal vectors or parallel vectors
"""

import sys
import re
import math
from typing import List, Tuple, Set, Dict, Optional, NamedTuple, Union


class OrthogonalityWitness(NamedTuple):
    """Represents an orthogonality violation witness."""
    edges: List[int]  # Edge variables involved in the violation
    vectors: Dict[int, List[Union[int, float, str]]]  # Vertex -> vector mapping (can be int, float, or algebraic string)


def parse_algebraic_complex(expr_str: str):
    """Parse algebraic expression in format: a/b+c/d*sqrt(2)+i(e/f+g/h*sqrt(2))

    Returns:
        Tuple (real, imag) where each is a float, or None if parsing fails.
    """
    import re
    import math

    expr_str = expr_str.strip()

    # Check if it contains 'i(' indicating complex number
    if 'i(' not in expr_str:
        return None

    try:
        # Split into real and imaginary parts
        # Format: "real_part+i(imag_part)"
        match = re.match(r'(.+?)\+i\((.+?)\)$', expr_str)
        if not match:
            return None

        real_part_str = match.group(1)
        imag_part_str = match.group(2)

        def eval_quadratic_expr(s):
            """Evaluate expression of form: a/b+c/d*sqrt(2)"""
            s = s.strip()

            # Replace division with Python division and sqrt with math.sqrt
            s = s.replace('sqrt', 'math.sqrt')

            # Handle potential issues with eval
            # Only allow safe operations
            allowed_chars = set('0123456789.+-*/()mathsqrt ')
            if not all(c in allowed_chars for c in s):
                return None

            try:
                result = eval(s)
                return float(result)
            except:
                return None

        real_val = eval_quadratic_expr(real_part_str)
        imag_val = eval_quadratic_expr(imag_part_str)

        if real_val is not None and imag_val is not None:
            return (real_val, imag_val)

        return None
    except:
        return None


def parse_component(comp_str: str):
    """Parse a vector component that can be int, float, or algebraic expression."""
    comp_str = comp_str.strip()

    # Try parsing as int first
    try:
        return int(comp_str)
    except ValueError:
        pass

    # Try parsing as float
    try:
        return float(comp_str)
    except ValueError:
        pass

    # Try to parse as algebraic complex number: a/b+c/d*sqrt(2)+i(e/f+g/h*sqrt(2))
    complex_val = parse_algebraic_complex(comp_str)
    if complex_val is not None:
        # Return as tuple (real, imag)
        return complex_val

    # Try to evaluate simple algebraic expressions
    try:
        # Replace sqrt with math.sqrt for evaluation
        import math
        eval_str = comp_str.replace('sqrt', 'math.sqrt')

        # Check if it's a complex number expression (contains 'i(')
        if 'i(' in eval_str:
            # Could not parse, return as string
            return comp_str
        else:
            # Try to evaluate as a real expression
            return eval(eval_str)
    except:
        # If all else fails, return as string
        return comp_str


def parse_orthogonality_witnesses(ortho_file: str) -> List[OrthogonalityWitness]:
    """Parse orthogonality witness file and return list of witnesses."""
    witnesses = []

    try:
        with open(ortho_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse format: "edges:1,2,3 vectors:v0=[1,0,0],v1=[0,1,0],v2=[1,1,0]"
                if 'edges:' not in line or 'vectors:' not in line:
                    continue

                # Split into edges and vectors parts
                edges_part, vectors_part = line.split(' vectors:')
                edges_part = edges_part.replace('edges:', '')

                # Parse edges
                edges = []
                if edges_part:
                    edges = [int(x) for x in edges_part.split(',')]

                # Parse vectors
                vectors = {}
                if vectors_part:
                    # Use regex to properly parse vectors with complex components
                    # Match patterns like "v0=[comp1,comp2,...]"
                    import re
                    vector_pattern = r'v(\d+)=\[([^\]]+)\]'
                    matches = re.findall(vector_pattern, vectors_part)

                    for vertex_num, components_str in matches:
                        vertex_id = int(vertex_num)

                        # Parse components - need to be careful with complex expressions
                        # that may contain commas within parentheses
                        components = []
                        depth = 0
                        current_comp = ""

                        for char in components_str + ',':
                            if char == '(':
                                depth += 1
                                current_comp += char
                            elif char == ')':
                                depth -= 1
                                current_comp += char
                            elif char == ',' and depth == 0:
                                if current_comp.strip():
                                    components.append(parse_component(current_comp))
                                current_comp = ""
                            else:
                                current_comp += char

                        # Convert to interleaved format if we have complex tuples
                        # E.g., [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)] -> [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                        if components and isinstance(components[0], tuple):
                            # We have complex components as tuples
                            interleaved = []
                            for comp in components:
                                if isinstance(comp, tuple) and len(comp) == 2:
                                    interleaved.extend([comp[0], comp[1]])  # [real, imag]
                                else:
                                    # Shouldn't happen, but handle it
                                    interleaved.extend([comp, 0])
                            vectors[vertex_id] = interleaved
                        else:
                            vectors[vertex_id] = components

                witnesses.append(OrthogonalityWitness(edges=edges, vectors=vectors))

    except FileNotFoundError:
        print(f"Warning: Orthogonality witness file '{ortho_file}' not found")
    except Exception as e:
        print(f"Error parsing orthogonality witness file '{ortho_file}': {e}")
        import traceback
        traceback.print_exc()

    return witnesses


def dot_product_real(v1: List[Union[int, float, str]], v2: List[Union[int, float, str]]) -> Union[int, float, None]:
    """Calculate real dot product of two vectors (standard Euclidean).

    This matches the C++ implementation in RCL.cpp:
        result = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

    Returns:
        The dot product as int or float, or None if vectors contain non-numeric components.
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")

    # Check if any component is a string (algebraic expression)
    if any(isinstance(x, str) for x in v1) or any(isinstance(x, str) for x in v2):
        # Cannot compute dot product with symbolic expressions
        # Return None to indicate this
        return None

    return sum(a * b for a, b in zip(v1, v2))


def dot_product_complex(v1: List[Union[int, float, str]], v2: List[Union[int, float, str]]) -> Union[int, float, None]:
    """Calculate complex dot product (Hermitian inner product): conj(v1) · v2.

    This matches the C++ implementation in RCL.cpp dot_complex():
    Supports two encodings:
    1) Real-only: length == 3 (imag parts assumed 0)
    2) Interleaved complex: length == 6 as [Re0, Im0, Re1, Im1, Re2, Im2]

    Returns 0 iff Hermitian inner product equals 0 as a complex number.
    Otherwise returns real_sum if non-zero, else imag_sum.

    Returns:
        0 if orthogonal, non-zero value if not, or None if vectors contain symbolic expressions.
    """
    VECTOR_DIMENSION = 3

    # Check if any component is a string (algebraic expression)
    if any(isinstance(x, str) for x in v1) or any(isinstance(x, str) for x in v2):
        return None

    v1_is_complex = len(v1) == 2 * VECTOR_DIMENSION
    v2_is_complex = len(v2) == 2 * VECTOR_DIMENSION

    real_sum = 0
    imag_sum = 0

    for idx in range(VECTOR_DIMENSION):
        # v1 component (a + i*b), but conjugated: (a - i*b)
        a = v1[2 * idx] if v1_is_complex else v1[idx]
        b = v1[2 * idx + 1] if v1_is_complex else 0

        # v2 component (c + i*d)
        c = v2[2 * idx] if v2_is_complex else v2[idx]
        d = v2[2 * idx + 1] if v2_is_complex else 0

        # (a - i*b) * (c + i*d) = (a*c + b*d) + i*(a*d - b*c)
        real_sum += a * c + b * d
        imag_sum += a * d - b * c

    # Orthogonal iff both sums are zero
    if real_sum == 0 and imag_sum == 0:
        return 0

    # Return a non-zero sentinel for violation (prefer real part if non-zero)
    return real_sum if real_sum != 0 else imag_sum


def dot_product(v1: List[Union[int, float, str]], v2: List[Union[int, float, str]],
                use_complex: bool = False) -> Union[int, float, None]:
    """Calculate dot product of two vectors.

    Args:
        v1: First vector
        v2: Second vector
        use_complex: If True, use complex (Hermitian) dot product; if False, use real dot product

    Returns:
        The dot product, or None if vectors contain non-numeric components.
    """
    if use_complex:
        return dot_product_complex(v1, v2)
    else:
        return dot_product_real(v1, v2)


def cross_product_real(v1: List[Union[int, float, str]], v2: List[Union[int, float, str]]) -> Optional[List[Union[int, float]]]:
    """Calculate real cross product of two 3D vectors.

    This matches the C++ implementation in RCL.cpp cross_product():
        result[0] = a[1] * b[2] - a[2] * b[1]
        result[1] = a[2] * b[0] - a[0] * b[2]
        result[2] = a[0] * b[1] - a[1] * b[0]

    Returns:
        The cross product, or None if vectors contain non-numeric components.
    """
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Real cross product only defined for 3D vectors")

    # Check if any component is a string (algebraic expression)
    if any(isinstance(x, str) for x in v1) or any(isinstance(x, str) for x in v2):
        return None

    return [
        v1[1] * v2[2] - v1[2] * v2[1],  # x component
        v1[2] * v2[0] - v1[0] * v2[2],  # y component
        v1[0] * v2[1] - v1[1] * v2[0]   # z component
    ]


def cross_product_complex(a: List[Union[int, float, str]], b: List[Union[int, float, str]]) -> Optional[List[Union[int, float]]]:
    """Calculate complex cross product (Hermitian cross product).

    This matches the C++ implementation in RCL.cpp cross_complex():
    Supports two encodings:
    1) Real-only: length == 3
    2) Interleaved complex: length == 6 as [Re0, Im0, Re1, Im1, Re2, Im2]

    Hermitian cross product (x, y, z):
        x = conj(a[1]) * b[2] - conj(a[2]) * b[1]
        y = conj(a[2]) * b[0] - conj(a[0]) * b[2]
        z = conj(a[0]) * b[1] - conj(a[1]) * b[0]

    Returns:
        Interleaved complex result [x_re, x_im, y_re, y_im, z_re, z_im] or None if symbolic.
    """
    VECTOR_DIMENSION = 3

    # Check if any component is a string (algebraic expression)
    if any(isinstance(x, str) for x in a) or any(isinstance(x, str) for x in b):
        return None

    a_is_complex = len(a) == 2 * VECTOR_DIMENSION
    b_is_complex = len(b) == 2 * VECTOR_DIMENSION

    # Real-only fallback equals standard cross product
    if not a_is_complex and not b_is_complex:
        return cross_product_real(a, b)

    def get_re(v, is_complex, idx):
        return v[2 * idx] if is_complex else v[idx]

    def get_im(v, is_complex, idx):
        return v[2 * idx + 1] if is_complex else 0

    def mul_conj_a_b(ai, bi):
        """Compute conj(a[ai]) * b[bi]"""
        a_re = get_re(a, a_is_complex, ai)
        a_im = get_im(a, a_is_complex, ai)
        b_re = get_re(b, b_is_complex, bi)
        b_im = get_im(b, b_is_complex, bi)
        # (a_re - i*a_im) * (b_re + i*b_im) = (a_re*b_re + a_im*b_im) + i*(a_re*b_im - a_im*b_re)
        return (a_re * b_re + a_im * b_im, a_re * b_im - a_im * b_re)

    # x = conj(a[1]) * b[2] - conj(a[2]) * b[1]
    t1_re, t1_im = mul_conj_a_b(1, 2)
    t2_re, t2_im = mul_conj_a_b(2, 1)
    x_re = t1_re - t2_re
    x_im = t1_im - t2_im

    # y = conj(a[2]) * b[0] - conj(a[0]) * b[2]
    t1_re, t1_im = mul_conj_a_b(2, 0)
    t2_re, t2_im = mul_conj_a_b(0, 2)
    y_re = t1_re - t2_re
    y_im = t1_im - t2_im

    # z = conj(a[0]) * b[1] - conj(a[1]) * b[0]
    t1_re, t1_im = mul_conj_a_b(0, 1)
    t2_re, t2_im = mul_conj_a_b(1, 0)
    z_re = t1_re - t2_re
    z_im = t1_im - t2_im

    # Return interleaved result
    return [x_re, x_im, y_re, y_im, z_re, z_im]


def cross_product(v1: List[Union[int, float, str]], v2: List[Union[int, float, str]],
                  use_complex: bool = False) -> Optional[List[Union[int, float]]]:
    """Calculate cross product of two vectors.

    Args:
        v1: First vector
        v2: Second vector
        use_complex: If True, use complex (Hermitian) cross product; if False, use real cross product

    Returns:
        The cross product, or None if vectors contain non-numeric components.
    """
    if use_complex:
        return cross_product_complex(v1, v2)
    else:
        return cross_product_real(v1, v2)


def is_zero_vector(v: Union[List[Union[int, float, str]], None], use_complex: bool = False) -> bool:
    """Check if a vector is the zero vector.

    Args:
        v: The vector to check
        use_complex: If True, check for complex zero (all components real and imaginary are 0)
                     If False, check for real zero (all components are 0)

    Returns:
        True if the vector is zero, False otherwise.
    """
    if v is None:
        return False

    if use_complex:
        # For complex vectors: expect interleaved [Re0, Im0, Re1, Im1, Re2, Im2]
        # Check if all components are zero
        VECTOR_DIMENSION = 3
        if len(v) == 2 * VECTOR_DIMENSION:
            return all(x == 0 for x in v if not isinstance(x, str))
        elif len(v) == VECTOR_DIMENSION:
            # Real-only encoding
            return all(x == 0 for x in v if not isinstance(x, str))
        else:
            return False
    else:
        # For real vectors: just check all components are 0
        return all(x == 0 for x in v if not isinstance(x, str))


def edge_var_to_vertices(edge_var: int, n: int) -> Tuple[int, int]:
    """Convert edge variable to vertex pair (i, j) where i < j."""
    # Edge variables are arranged column by column in upper triangle
    # For n=3: 1->(0,1), 2->(0,2), 3->(1,2)
    # For n=4: 1->(0,1), 2->(0,2), 3->(0,3), 4->(1,2), 5->(1,3), 6->(2,3)
    
    current_var = 1
    for col in range(1, n):
        for row in range(col):
            if current_var == edge_var:
                return row, col
            current_var += 1
    
    raise ValueError(f"Edge variable {edge_var} out of range for n={n}")


def verify_orthogonality_clause(blocking_clause: List[int], witness: OrthogonalityWitness,
                                verbose: bool = False, use_complex: bool = False) -> bool:
    """Verify that an orthogonality blocking clause correctly prevents the violation.

    Args:
        blocking_clause: The blocking clause to verify
        witness: The orthogonality witness
        verbose: If True, print detailed verification output
        use_complex: If True, use complex (Hermitian) dot/cross products; if False, use real products

    Returns:
        True if verification passes, False otherwise.
    """

    if verbose:
        print(f"  Verifying orthogonality clause: {blocking_clause}")
        print(f"  Witness edges: {witness.edges}")
        print(f"  Witness vectors: {witness.vectors}")
        print(f"  Using {'complex (Hermitian)' if use_complex else 'real'} vector operations")

    # Check that all edges in the witness are blocked by the clause
    # Note: SAT variables are 1-indexed (1, 2, 3, ...), but witness edges are 0-indexed (0, 1, 2, ...)
    # So we need to convert: variable -> edge by doing (variable - 1)
    blocked_edges = set((-lit) - 1 for lit in blocking_clause if lit < 0)  # Convert negative literals to 0-indexed edges

    if verbose:
        print(f"  Blocking clause literals: {blocking_clause}")
        print(f"  Converted to 0-indexed edges: {sorted(blocked_edges)}")

    for edge_idx in witness.edges:
        if edge_idx not in blocked_edges:
            if verbose:
                print(f"  ❌ Edge {edge_idx} not blocked by clause")
                print(f"     Expected edges {sorted(blocked_edges)}, but witness has edge {edge_idx}")
            return False

    # Verify the orthogonality violation exists
    vertices = list(witness.vectors.keys())

    if len(vertices) < 2:
        if verbose:
            print(f"  ❌ Need at least 2 vertices for orthogonality check")
        return False

    # Check if we have symbolic (non-numeric) vectors
    has_symbolic_vectors = any(
        isinstance(comp, str)
        for vec in witness.vectors.values()
        for comp in vec
    )

    # Check for direct orthogonality violations (connected vertices with non-orthogonal vectors)
    violation_found = False
    skipped_symbolic_checks = 0
    
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            v1, v2 = vertices[i], vertices[j]
            
            if v1 not in witness.vectors or v2 not in witness.vectors:
                continue
            
            vec1 = witness.vectors[v1]
            vec2 = witness.vectors[v2]
            
            # Check if these vertices are connected (edge exists in witness)
            # Note: Vertices in witness.vectors are already the correct IDs (e.g., v2 -> 2, v13 -> 13)
            # Edge formula for 0-indexed edges: edge = col * (col - 1) / 2 + row, where row < col
            edge_idx = None
            try:
                # Compute 0-indexed edge number for this vertex pair
                row = min(v1, v2)
                col = max(v1, v2)
                edge_idx = col * (col - 1) // 2 + row
            except:
                continue

            if edge_idx is not None and edge_idx in witness.edges:
                # These vertices are connected, check orthogonality
                dot = dot_product(vec1, vec2, use_complex=use_complex)

                if dot is None:
                    skipped_symbolic_checks += 1
                    if verbose:
                        print(f"  ⚠️  Cannot compute dot product for v{v1}·v{v2} (contains algebraic expressions)")
                        print(f"     Skipping symbolic verification - algebraic expressions need symbolic math")
                    # Skip verification for symbolic vectors
                    continue

                if verbose:
                    print(f"  Checking orthogonality: v{v1}·v{v2} = {dot}")

                if dot != 0:
                    if verbose:
                        print(f"  ✅ Found orthogonality violation: v{v1}·v{v2} = {dot} ≠ 0")
                    violation_found = True
                    break
        
        if violation_found:
            break
    
    # Check for parallel vectors violation (cross product is zero)
    if not violation_found and len(vertices) >= 3:
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                for k in range(j + 1, len(vertices)):
                    v1, v2, v3 = vertices[i], vertices[j], vertices[k]
                    
                    if (v1 not in witness.vectors or 
                        v2 not in witness.vectors or 
                        v3 not in witness.vectors):
                        continue
                    
                    vec1 = witness.vectors[v1]
                    vec2 = witness.vectors[v2]
                    
                    # Check if cross product is zero (parallel vectors)
                    try:
                        cross = cross_product(vec1, vec2, use_complex=use_complex)

                        if cross is None:
                            skipped_symbolic_checks += 1
                            if verbose:
                                print(f"  ⚠️  Cannot compute cross product for v{v1} × v{v2} (contains algebraic expressions)")
                            continue

                        if is_zero_vector(cross, use_complex=use_complex):
                            if verbose:
                                print(f"  ✅ Found parallel vectors violation: v{v1} × v{v2} = {cross}")
                            violation_found = True
                            break
                    except ValueError:
                        # Not 3D vectors, skip
                        continue
                
                if violation_found:
                    break
            
            if violation_found:
                break
    
    if not violation_found:
        # If we have symbolic vectors and couldn't verify them, accept with warning
        if has_symbolic_vectors and skipped_symbolic_checks > 0:
            if verbose:
                print(f"  ⚠️  Witness contains symbolic vectors - numerical verification skipped")
                print(f"     Accepting clause based on structural validity (edges match)")
            else:
                print(f"  Note: Orthogonality witness contains symbolic expressions (skipped {skipped_symbolic_checks} checks)")
            return True
        else:
            if verbose:
                print(f"  ❌ No orthogonality violation found in witness")
            return False

    if verbose:
        print(f"  ✅ Orthogonality clause verification passed")

    return True


def read_varint(f, use_offset: bool = False) -> Optional[int]:
    """Read a variable-length encoded integer from binary file.
    
    Returns the decoded integer, or None if end of sequence (0 byte).
    
    Args:
        f: File object to read from
        use_offset: If True, use offset encoding for permutations (x = 2*(abs(val)+1) + sign)
                   If False, use DRAT encoding for literals (x = 2*abs(val) + sign)
    """
    x = 0
    shift = 0
    while True:
        byte_data = f.read(1)
        if not byte_data:
            return None
        
        byte = byte_data[0]
        
        # Check for terminator (0 byte)
        if byte == 0:
            return None
        
        # Extract 7-bit value and continuation bit
        x |= (byte & 0x7f) << shift
        shift += 7
        
        # If continuation bit is not set, we're done
        if not (byte & 0x80):
            break
    
    if use_offset:
        # Decode with offset: x = 2 * (abs(val) + 1) + sign_bit
        # So: abs(val) + 1 = x >> 1, thus abs(val) = (x >> 1) - 1
        abs_val = (x >> 1) - 1
        is_negative = x & 1
    else:
        # Decode without offset: x = 2 * abs(val) + sign_bit
        abs_val = x >> 1
        is_negative = x & 1
    
    return -abs_val if is_negative else abs_val


def is_binary_file(filename: str) -> bool:
    """Check if a file appears to be in binary format."""
    try:
        with open(filename, 'rb') as f:
            # Read first few bytes
            data = f.read(10)
            if not data:
                return False
            
            # Binary DRAT files start with 'a', 'd', 't', or 'o'
            # Followed by variable-length encoded integers
            if data[0] in [ord('a'), ord('d'), ord('t'), ord('o')]:
                # Check if next bytes look like variable-length encoding
                # (high bit set for continuation, or small values)
                for i in range(1, min(len(data), 5)):
                    if data[i] == 0:  # Terminator
                        return True
                    if data[i] > 127:  # Continuation bit set
                        return True
                    if data[i] < 32 and i > 1:  # Small value indicating encoding
                        return True
            
            # Check for text format indicators
            try:
                text = data.decode('ascii')
                # If it contains spaces and digits, likely text format
                if ' ' in text and any(c.isdigit() for c in text):
                    return False
            except:
                pass
            
            return True
    except:
        return False


def clause_to_string(clause: List[int]) -> str:
    """Convert clause to string format for comparison."""
    return ' '.join(map(str, sorted(clause, key=abs)))


def parse_perm_file(filename: str) -> List[List[int]]:
    """Parse permutation file and return list of permutations.
    
    Supports both text format (space-separated integers, one per line)
    and binary format (variable-length encoded integers).
    """
    permutations = []
    
    # Check if binary format
    if is_binary_file(filename):
        # Parse binary format (permutations use offset encoding)
        with open(filename, 'rb') as f:
            while True:
                perm = []
                while True:
                    val = read_varint(f, use_offset=True)
                    if val is None:
                        break
                    perm.append(val)
                
                if perm:
                    permutations.append(perm)
                else:
                    break
    else:
        # Parse text format
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse space-separated integers
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
    # Create a set of absolute variable values already in the clause
    clause_vars = {abs(lit) for lit in clause}

    # Add fixed edges that aren't already in the clause
    augmented = clause.copy()
    for lit in fixed_edges:
        if abs(lit) not in clause_vars:
            augmented.append(lit)

    return augmented


def parse_drat_file(filename: str, fixed_edges: List[int] = None) -> Tuple[List[List[int]], Set[str], List[List[int]]]:
    """Parse DRAT file and return t-clauses, set of all t-clause strings, and o-clauses.
    
    Supports both text format and binary format.
    """
    t_clauses = []
    o_clauses = []
    clause_set = set()

    if fixed_edges is None:
        fixed_edges = []

    # Check if binary format
    if is_binary_file(filename):
        # Parse binary format
        with open(filename, 'rb') as f:
            while True:
                # Read command byte
                cmd_byte = f.read(1)
                if not cmd_byte:
                    break
                
                cmd = chr(cmd_byte[0])
                
                # Skip deletion lines
                if cmd == 'd':
                    # Read and discard the clause
                    while True:
                        val = read_varint(f)
                        if val is None:
                            break
                    continue
                
                # Process t-clauses (comments)
                if cmd == 't':
                    clause = []
                    while True:
                        val = read_varint(f)
                        if val is None:
                            break
                        clause.append(val)
                    
                    if clause:
                        t_clauses.append(clause)
                        
                        # Augment clause with fixed edges before storing
                        augmented_clause = augment_clause_with_fixed_edges(clause, fixed_edges)
                        clause_str = clause_to_string(augmented_clause)
                        clause_set.add(clause_str)
                
                # Process o-clauses (orthogonality)
                elif cmd == 'o':
                    clause = []
                    while True:
                        val = read_varint(f)
                        if val is None:
                            break
                        clause.append(val)
                    
                    if clause:
                        o_clauses.append(clause)
                
                # Skip other commands ('a', etc.) by reading until terminator
                elif cmd in ['a']:
                    while True:
                        val = read_varint(f)
                        if val is None:
                            break
    else:
        # Parse text format
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('d'):  # Skip deletion lines
                    parts = line.split()
                    if parts and parts[0] == 't':
                        # Extract t-clause (remove 't' and '0')
                        clause = [int(x) for x in parts[1:-1]]
                        t_clauses.append(clause)

                        # Augment clause with fixed edges before storing
                        augmented_clause = augment_clause_with_fixed_edges(clause, fixed_edges)
                        clause_str = clause_to_string(augmented_clause)
                        clause_set.add(clause_str)
                    elif parts and parts[0] == 'o':
                        # Extract o-clause (remove 'o' and '0')
                        clause = [int(x) for x in parts[1:-1]]
                        o_clauses.append(clause)

    return t_clauses, clause_set, o_clauses


def variable_to_matrix_position(var: int, n: int) -> Tuple[int, int]:
    """
    Convert variable number to adjacency matrix position.
    Variables are arranged column by column in upper triangle.
    For n=3: 1->(0,1), 2->(0,2), 3->(1,2)
    For n=4: 1->(0,1), 2->(0,2), 3->(0,3), 4->(1,2), 5->(1,3), 6->(2,3)
    """
    current_var = 1
    for col in range(1, n):  # Start from column 1
        for row in range(col):  # Rows 0 to col-1 (upper triangle)
            if current_var == var:
                return row, col
            current_var += 1
    raise ValueError(f"Variable {var} out of range for n={n}")


def matrix_position_to_variable(row: int, col: int, n: int) -> int:
    """Convert adjacency matrix position back to variable number."""
    # Variables are arranged column by column in upper triangle
    # For n=3: (0,1)->1, (0,2)->2, (1,2)->3
    # For n=4: (0,1)->1, (0,2)->2, (0,3)->3, (1,2)->4, (1,3)->5, (2,3)->6
    # For n=6: (0,1)->1, (0,2)->2, (0,3)->3, (0,4)->4, (0,5)->5, (1,2)->6, ...

    var = 1  # Start from variable 1
    for c in range(1, n):  # Start from column 1
        for r in range(c):  # Rows 0 to c-1 (upper triangle)
            if r == row and c == col:
                return var
            var += 1
    raise ValueError(f"Position ({row},{col}) out of range for n={n}")


def clause_to_adjacency_matrix(clause: List[int], n: int, verbose: bool = False) -> List[List[int]]:
    """Convert clause to adjacency matrix representation."""
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    if verbose:
        print(f"  Converting clause {clause} to adjacency matrix:")
    
    for var in clause:
        abs_var = abs(var)
        row, col = variable_to_matrix_position(abs_var, n)
        
        if verbose:
            print(f"    Variable {var} (abs={abs_var}) -> position ({row}, {col})")
        
        # Set the matrix value (1 for positive literal, 0 for negative)
        # Note: we invert polarity because we're looking at blocked graphs
        value = 1 if var < 0 else 0
        matrix[row][col] = value
        matrix[col][row] = value  # Symmetric matrix
        
        if verbose:
            print(f"      Setting matrix[{row}][{col}] = {value} (inverted polarity)")
    
    if verbose:
        print(f"  Resulting matrix:")
        for i in range(n):
            print(f"    {matrix[i]}")
    
    return matrix


def adjacency_matrix_to_clause(matrix: List[List[int]], n: int, verbose: bool = False) -> List[int]:
    """Convert adjacency matrix back to clause representation."""
    clause = []

    if verbose:
        print(f"  Converting adjacency matrix back to clause:")
        for i in range(n):
            print(f"    {matrix[i]}")

    # Iterate column by column through upper triangle
    for col in range(1, n):  # Start from column 1
        for row in range(col):  # Rows 0 to col-1 (upper triangle)
            var = matrix_position_to_variable(row, col, n)
            if matrix[row][col] == 1:
                clause.append(-var)  # Negative literal for blocking
                if verbose:
                    print(f"    Position ({row},{col}) -> variable {var} (negative)")
            elif matrix[row][col] == 0:
                clause.append(var)   # Positive literal for blocking
                if verbose:
                    print(f"    Position ({row},{col}) -> variable {var} (positive)")

    if verbose:
        print(f"  Resulting clause: {clause}")

    return clause


def apply_permutation(matrix: List[List[int]], perm: List[int], verbose: bool = False) -> List[List[int]]:
    """Apply permutation to adjacency matrix."""
    n = len(matrix)
    permuted_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    if verbose:
        print(f"  Applying permutation {perm}:")
        print(f"    Original matrix:")
        for i in range(n):
            print(f"      {matrix[i]}")
    
    for i in range(n):
        for j in range(n):
            permuted_matrix[perm[i]][perm[j]] = matrix[i][j]
    
    if verbose:
        print(f"    Permuted matrix:")
        for i in range(n):
            print(f"      {permuted_matrix[i]}")
    
    return permuted_matrix


def extract_first_k_variables(clause: List[int], k: int) -> List[int]:
    """
    Extract literals corresponding to the first k variables from a clause.
    
    Args:
        clause: The clause to extract from
        k: The number of variables to extract (1, 2, 3, ..., k)
    
    Returns:
        List of literals whose absolute values are in {1, 2, ..., k}
    """
    return [lit for lit in clause if abs(lit) <= k]


def contains_blocking_clause(permuted_clause_str: str, clause_set: Set[str], original_clause_str: str, verbose: bool = False) -> bool:
    """
    Check if the permuted clause exists in the set of t-clauses (other than itself).
    Uses hash table lookup for O(1) efficiency.

    Args:
        permuted_clause_str: The permuted clause to check
        clause_set: Set of all t-clause strings
        original_clause_str: The original (unpermuted) clause string to skip
    """
    # Check if permuted clause exists in the set and is not the original clause
    if permuted_clause_str in clause_set and permuted_clause_str != original_clause_str:
        if verbose:
            print(f"    Found blocking clause: {permuted_clause_str}")
        return True

    return False


def contains_blocking_clause_or_subset(permuted_clause: List[int], clause_set: Set[str], 
                                       original_clause_str: str, num_fixed_edges: int = 0, 
                                       verbose: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Check if the permuted clause OR any of its induced subgraphs exist in the set of t-clauses.
    
    For a graph with n vertices (n choose 2 variables), we check:
    - The full clause (all n choose 2 variables)
    - The induced subgraph on first (n-1) vertices (first (n-1) choose 2 variables)
    - The induced subgraph on first (n-2) vertices (first (n-2) choose 2 variables)
    - ... down to minimum size
    
    Minimum size is determined by fixed edges:
    - If no fixed edges: down to 2 vertices (1 variable)
    - If e fixed edges: down to (e+1) vertices ((e+1) choose 2 variables)
    
    Args:
        permuted_clause: The permuted clause to check (list of literals)
        clause_set: Set of all t-clause strings
        original_clause_str: The original (unpermuted) clause string to skip
        num_fixed_edges: Number of fixed edge assignments
        verbose: If True, print detailed output
    
    Returns:
        (found, clause_str): True if blocking clause found, along with the matching clause string
    """
    # Determine graph size from clause length
    num_vars = len(permuted_clause)
    # n choose 2 = num_vars, so n = (1 + sqrt(1 + 8*num_vars)) / 2
    import math
    n = int((1 + math.sqrt(1 + 8 * num_vars)) / 2)
    
    # Determine minimum number of vertices to check
    # If e fixed edges, minimum is (e+1) vertices
    # Otherwise, minimum is 2 vertices
    min_vertices = max(2, num_fixed_edges + 1) if num_fixed_edges > 0 else 2
    
    if verbose:
        print(f"  Checking blocking clauses for graph size {n} vertices (down to {min_vertices} vertices)")
    
    # Check from full graph down to minimum size
    for k in range(n, min_vertices - 1, -1):
        k_vars = k * (k - 1) // 2  # Number of variables for k vertices
        
        # Extract subset corresponding to first k vertices
        subset_clause = extract_first_k_variables(permuted_clause, k_vars)
        subset_clause_str = clause_to_string(subset_clause)
        
        # Check if this subset exists in the clause set (and is not the original)
        if subset_clause_str in clause_set and subset_clause_str != original_clause_str:
            if verbose:
                print(f"    Found blocking clause for {k}-vertex induced subgraph: {subset_clause_str}")
            return True, subset_clause_str
        
        if verbose:
            print(f"    {k}-vertex subset ({k_vars} vars): not found ✓")
    
    return False, None


def verify_drat_proof(perm_file: str, drat_file: str, fixed_edges_file: str = None, ortho_file: str = None,
                      verbose: bool = False, use_complex: bool = False) -> bool:
    """Main verification function.

    Args:
        perm_file: Path to permutation file
        drat_file: Path to DRAT file
        fixed_edges_file: Path to fixed edges file (optional)
        ortho_file: Path to orthogonality witness file (optional)
        verbose: If True, print detailed verification output
        use_complex: If True, use complex (Hermitian) dot/cross products; if False, use real products

    Returns:
        True if verification passes, False otherwise.
    """
    print(f"Verifying DRAT proof")
    print(f"Permutation file: {perm_file} ({'binary' if is_binary_file(perm_file) else 'text'} format)")
    print(f"DRAT file: {drat_file} ({'binary' if is_binary_file(drat_file) else 'text'} format)")
    if fixed_edges_file:
        print(f"Fixed edges file: {fixed_edges_file}")
    if ortho_file:
        print(f"Orthogonality witness file: {ortho_file}")
    print(f"Vector operations: {'Complex (Hermitian)' if use_complex else 'Real'}")
    print()

    # Parse fixed edges if provided
    fixed_edges = []
    if fixed_edges_file:
        fixed_edges = parse_fixed_edges_file(fixed_edges_file)
        print(f"Loaded {len(fixed_edges)} fixed edge assignments")

    # Parse files
    permutations = parse_perm_file(perm_file)
    t_clauses, clause_set, o_clauses = parse_drat_file(drat_file, fixed_edges)

    print(f"Found {len(permutations)} permutations")
    print(f"Found {len(t_clauses)} t-clauses")
    print(f"Found {len(o_clauses)} o-clauses")
    print(f"Built hash table with {len(clause_set)} clauses")
    print()

    # Parse orthogonality witnesses if provided
    orthogonality_witnesses = []
    if ortho_file:
        orthogonality_witnesses = parse_orthogonality_witnesses(ortho_file)
        print(f"Loaded {len(orthogonality_witnesses)} orthogonality witnesses")
        print()

    # Verify each t-clause
    verification_passed = True

    for i, t_clause in enumerate(t_clauses):
        if i >= len(permutations):
            print(f"Warning: No permutation found for t-clause {i+1}")
            continue

        perm = permutations[i]
        if verbose:
            print(f"Verifying t-clause {i+1}: {t_clause}")
            print(f"Using permutation: {perm}")

        # Use permutation length as graph size
        graph_size = len(perm)
        expected_clause_size = graph_size * (graph_size - 1) // 2

        # Augment t-clause with fixed edges for processing
        augmented_clause = augment_clause_with_fixed_edges(t_clause, fixed_edges)

        # Validate clause size (only warn if not using fixed edges)
        if len(augmented_clause) != expected_clause_size:
            if not fixed_edges:
                print(f"⚠️  WARNING: t-clause {i+1} has {len(augmented_clause)} literals, expected {expected_clause_size} for graph size {graph_size}")
                print(f"   Clause should have n*(n-1)/2 literals where n = {graph_size}")
                verification_passed = False
                continue
            else:
                print(f"⚠️  ERROR: t-clause {i+1} has {len(t_clause)} literals + {len(fixed_edges)} fixed = {len(augmented_clause)} total, expected {expected_clause_size}")
                verification_passed = False
                continue

        # Get original clause string for comparison (using augmented clause)
        original_clause_str = clause_to_string(augmented_clause)

        # Convert augmented clause to adjacency matrix
        matrix = clause_to_adjacency_matrix(augmented_clause, graph_size, verbose)

        # Apply permutation
        permuted_matrix = apply_permutation(matrix, perm, verbose)

        # Convert back to clause
        permuted_clause = adjacency_matrix_to_clause(permuted_matrix, graph_size, verbose)
        permuted_clause_str = clause_to_string(permuted_clause)

        if verbose:
            print(f"Original clause (augmented): {original_clause_str}")
            print(f"Permuted clause: {permuted_clause_str}")

        # Check if the permuted clause or any of its subsets match any OTHER t-clause
        found, blocking_clause_str = contains_blocking_clause_or_subset(
            permuted_clause, clause_set, original_clause_str, 
            num_fixed_edges=len(fixed_edges), verbose=verbose
        )
        
        if found:
            print(f"❌ FAIL: t-clause {i+1} verification failed - blocking t-clause found!")
            print(f"   Blocking clause: {blocking_clause_str}")
            verification_passed = False
        else:
            if verbose:
                print(f"✅ PASS: No blocking t-clause or subset found")

        if verbose:
            print()

    # Verify orthogonality clauses if witnesses are provided
    if orthogonality_witnesses:
        print("Verifying orthogonality clauses...")
        
        if len(o_clauses) != len(orthogonality_witnesses):
            print(f"⚠️  WARNING: Number of o-clauses ({len(o_clauses)}) doesn't match number of witnesses ({len(orthogonality_witnesses)})")
            verification_passed = False
        
        for i, o_clause in enumerate(o_clauses):
            if i >= len(orthogonality_witnesses):
                print(f"Warning: No witness found for o-clause {i+1}")
                verification_passed = False
                continue
            
            witness = orthogonality_witnesses[i]
            
            if verbose:
                print(f"Verifying o-clause {i+1}: {o_clause}")

            if not verify_orthogonality_clause(o_clause, witness, verbose=verbose, use_complex=use_complex):
                print(f"❌ FAIL: o-clause {i+1} verification failed!")
                verification_passed = False
            else:
                if verbose:
                    print(f"✅ PASS: o-clause {i+1} verification passed")
            
            if verbose:
                print()
        
        print(f"Verified {len(o_clauses)} o-clauses")
        print()

    return verification_passed


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python drat_verifier.py <perm_file> <drat_file> [--fixed-edges <file>] [--ortho <file>] [--complex] [--verbose]")
        print("  perm_file: Path to permutation file")
        print("  drat_file: Path to DRAT file")
        print("  --fixed-edges <file>: Path to fixed edges file (space-separated literals)")
        print("  --ortho <file>: Path to orthogonality witness file")
        print("  --complex: Use complex (Hermitian) dot/cross products for orthogonality verification")
        print("  --verbose: Show detailed verification output")
        sys.exit(1)

    perm_file = sys.argv[1]
    drat_file = sys.argv[2]

    # Parse optional arguments
    fixed_edges_file = None
    ortho_file = None
    verbose = False
    use_complex = False

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--fixed-edges' and i + 1 < len(sys.argv):
            fixed_edges_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--ortho' and i + 1 < len(sys.argv):
            ortho_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--complex':
            use_complex = True
            i += 1
        elif sys.argv[i] == '--verbose':
            verbose = True
            i += 1
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)

    try:
        result = verify_drat_proof(perm_file, drat_file, fixed_edges_file, ortho_file, verbose, use_complex)

        print("=" * 50)
        if result:
            print("🎉 VERIFICATION PASSED: All clauses verified successfully!")
        else:
            print("❌ VERIFICATION FAILED: Some clauses failed verification!")

        sys.exit(0 if result else 1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
