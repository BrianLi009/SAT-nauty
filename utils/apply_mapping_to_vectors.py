#!/usr/bin/env python3
"""
Script to apply vertex mapping to reorder vectors and output in accepted format.

Usage:
    python3 apply_mapping_to_vectors.py --mapping <mapping_file> --vectors <vectors_dict> --output <output_file>
    python3 apply_mapping_to_vectors.py --interactive

Example:
    python3 apply_mapping_to_vectors.py --mapping graph_002_mapping.txt --vectors vectors_dict.py --output graph_002_vectors.txt
"""

import argparse
import sys
import os
import ast
import math
from pathlib import Path

def parse_mapping_file(mapping_file):
    """
    Parse the mapping file to extract vertex mappings.
    Returns a dictionary mapping original -> canonical vertex indices.
    """
    mapping = {}
    
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip header lines and empty lines
                if not line or "=" in line or "Mapping" in line:
                    continue
                    
                if "->" in line and "Vertex" in line:
                    # Parse lines like "Vertex 0 -> Vertex 13"
                    parts = line.split("->")
                    if len(parts) == 2:
                        original_part = parts[0].strip()
                        canonical_part = parts[1].strip()
                        
                        # Extract vertex numbers
                        original_vertex = int(original_part.replace("Vertex", "").strip())
                        canonical_vertex = int(canonical_part.replace("Vertex", "").strip())
                        
                        mapping[original_vertex] = canonical_vertex
        
        return mapping
        
    except Exception as e:
        print(f"Error parsing mapping file: {e}")
        return None

def parse_vectors_dict(vectors_input):
    """
    Parse vectors from either a file containing a Python dictionary or a direct dictionary.
    """
    if isinstance(vectors_input, str):
        # Try to parse as a file path first
        if os.path.exists(vectors_input):
            try:
                with open(vectors_input, 'r') as f:
                    content = f.read()
                
                # If it's a Python file with variables, try to extract the dictionary
                if 'vectors_dict' in content:
                    # Create a local namespace and execute the file
                    local_vars = {}
                    exec(content, {}, local_vars)
                    if 'vectors_dict' in local_vars:
                        return local_vars['vectors_dict']
                
                # Try to parse as literal first (works for simple numeric dictionaries)
                # Skip if content contains sqrt() or other function calls
                if 'sqrt(' in content:
                    # Skip ast.literal_eval for files with symbolic expressions
                    pass
                else:
                    try:
                        return ast.literal_eval(content)
                    except Exception:
                        # If literal_eval fails for any reason, try manual parsing
                        pass
                
                # Manual parsing to handle symbolic expressions like sqrt(2)
                import re
                import math
                
                # Replace sqrt with math.sqrt for evaluation, but preserve structure
                # We'll parse manually to keep symbolic expressions as strings
                vectors_dict = {}
                
                # Remove outer braces and whitespace
                content_clean = content.strip()
                if content_clean.startswith('{') and content_clean.endswith('}'):
                    content_clean = content_clean[1:-1].strip()
                
                # Split by top-level commas (not inside parentheses)
                items = []
                current_item = ""
                paren_count = 0
                
                for char in content_clean:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == ',' and paren_count == 0:
                        items.append(current_item.strip())
                        current_item = ""
                        continue
                    current_item += char
                
                if current_item.strip():
                    items.append(current_item.strip())
                
                # Parse each item
                for item in items:
                    if ':' in item:
                        key_part, value_part = item.split(':', 1)
                        key = int(key_part.strip())
                        
                        # Parse the tuple value
                        value_part = value_part.strip()
                        if value_part.startswith('(') and value_part.endswith(')'):
                            value_part = value_part[1:-1]
                        
                        # Split by commas, preserving symbolic expressions
                        coords = []
                        current_coord = ""
                        paren_count = 0
                        
                        for char in value_part:
                            if char == '(':
                                paren_count += 1
                            elif char == ')':
                                paren_count -= 1
                            elif char == ',' and paren_count == 0:
                                coords.append(current_coord.strip())
                                current_coord = ""
                                continue
                            current_coord += char
                        
                        if current_coord.strip():
                            coords.append(current_coord.strip())
                        
                        # Convert coordinates, preserving symbolic expressions
                        coord_values = []
                        for coord in coords:
                            coord = coord.strip()
                            if coord == 'I':
                                coord_values.append(1j)
                            elif coord == '-I':
                                coord_values.append(-1j)
                            elif 'sqrt(' in coord or 'I' in coord or '*' in coord or '/' in coord:
                                # Keep as string for symbolic expressions
                                coord_values.append(coord)
                            else:
                                # Try to convert to number
                                try:
                                    if '/' in coord:
                                        # Handle fractions
                                        num, den = coord.split('/')
                                        coord_values.append(float(num.strip()) / float(den.strip()))
                                    else:
                                        coord_values.append(float(coord))
                                except:
                                    coord_values.append(coord)
                        
                        vectors_dict[key] = tuple(coord_values)
                
                return vectors_dict if vectors_dict else None
            except Exception as e:
                print(f"Error reading vectors file: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # Try to parse as a dictionary string
            try:
                import re
                
                # Parse the dictionary string manually to preserve symbolic expressions
                vectors_dict = {}
                
                # Remove outer braces
                content = vectors_input.strip()
                if content.startswith('{') and content.endswith('}'):
                    content = content[1:-1]
                
                # Split by commas, but be careful with nested parentheses
                items = []
                current_item = ""
                paren_count = 0
                
                for char in content:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == ',' and paren_count == 0:
                        items.append(current_item.strip())
                        current_item = ""
                        continue
                    current_item += char
                
                if current_item.strip():
                    items.append(current_item.strip())
                
                # Parse each item
                for item in items:
                    if ':' in item:
                        key_part, value_part = item.split(':', 1)
                        key = int(key_part.strip())
                        
                        # Parse the tuple value
                        value_part = value_part.strip()
                        if value_part.startswith('(') and value_part.endswith(')'):
                            value_part = value_part[1:-1]
                        
                        # Split by commas, preserving symbolic expressions
                        coords = []
                        current_coord = ""
                        paren_count = 0
                        
                        for char in value_part:
                            if char == '(':
                                paren_count += 1
                            elif char == ')':
                                paren_count -= 1
                            elif char == ',' and paren_count == 0:
                                coords.append(current_coord.strip())
                                current_coord = ""
                                continue
                            current_coord += char
                        
                        if current_coord.strip():
                            coords.append(current_coord.strip())
                        
                        # Convert coordinates, preserving symbolic expressions
                        coord_values = []
                        for coord in coords:
                            coord = coord.strip()
                            if coord == 'I':
                                coord_values.append(1j)
                            elif coord == '-I':
                                coord_values.append(-1j)
                            elif 'sqrt(' in coord or 'I' in coord or '*' in coord:
                                # Keep as string for symbolic expressions (including complex ones)
                                coord_values.append(coord)
                            else:
                                # Try to convert to number
                                try:
                                    if '/' in coord:
                                        # Handle fractions
                                        num, den = coord.split('/')
                                        coord_values.append(float(num.strip()) / float(den.strip()))
                                    else:
                                        coord_values.append(float(coord))
                                except:
                                    coord_values.append(coord)
                        
                        vectors_dict[key] = tuple(coord_values)
                
                return vectors_dict
                
            except Exception as e:
                print(f"Error parsing vectors as dictionary: {e}")
                return None
    else:
        # Already a dictionary
        return vectors_input

def apply_mapping_to_vectors(vectors_dict, mapping):
    """
    Apply the mapping to reorder vectors according to canonical vertex order.
    Returns a list of tuples in canonical vertex order.
    """
    # Create a list to hold vectors in canonical order
    max_canonical_vertex = max(mapping.values())
    canonical_vectors = [None] * (max_canonical_vertex + 1)
    
    # Apply mapping: original_vertex -> canonical_vertex
    for original_vertex, canonical_vertex in mapping.items():
        if original_vertex in vectors_dict:
            canonical_vectors[canonical_vertex] = vectors_dict[original_vertex]
    
    # Filter out None values and return
    return [v for v in canonical_vectors if v is not None]

def format_vector(vector_tuple):
    """
    Format a vector tuple to string format, preserving symbolic expressions.
    """
    formatted_coords = []
    for coord in vector_tuple:
        if isinstance(coord, str):
            # Already a symbolic expression
            formatted_coords.append(coord)
        elif isinstance(coord, complex):
            # Handle complex numbers
            if coord.imag == 0:
                formatted_coords.append(str(int(coord.real)) if coord.real.is_integer() else str(coord.real))
            elif coord.real == 0:
                if coord.imag == 1:
                    formatted_coords.append("I")
                elif coord.imag == -1:
                    formatted_coords.append("-I")
                else:
                    formatted_coords.append(f"{coord.imag}*I")
            else:
                if coord.imag == 1:
                    formatted_coords.append(f"{coord.real}+I")
                elif coord.imag == -1:
                    formatted_coords.append(f"{coord.real}-I")
                else:
                    formatted_coords.append(f"{coord.real}+{coord.imag}*I")
        else:
            # Handle regular numbers
            if isinstance(coord, float) and coord.is_integer():
                formatted_coords.append(str(int(coord)))
            else:
                formatted_coords.append(str(coord))
    
    return " ".join(formatted_coords)

def process_vectors_with_mapping(mapping_file, vectors_input, output_file):
    """
    Main function to process vectors with mapping and output in accepted format.
    """
    # Parse mapping file
    print(f"Parsing mapping file: {mapping_file}")
    mapping = parse_mapping_file(mapping_file)
    if mapping is None:
        print("Failed to parse mapping file")
        return False
    
    print(f"Found {len(mapping)} vertex mappings")
    
    # Parse vectors
    print(f"Parsing vectors from: {vectors_input}")
    vectors_dict = parse_vectors_dict(vectors_input)
    if vectors_dict is None:
        print("Failed to parse vectors")
        return False
    
    print(f"Found {len(vectors_dict)} vectors")
    
    # Apply mapping
    print("Applying mapping to reorder vectors...")
    canonical_vectors = apply_mapping_to_vectors(vectors_dict, mapping)
    
    print(f"Generated {len(canonical_vectors)} canonical vectors")
    
    # Write output file
    print(f"Writing output to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            for vector in canonical_vectors:
                formatted_vector = format_vector(vector)
                f.write(formatted_vector + '\n')
        
        print("Successfully created output file")
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Apply vertex mapping to reorder vectors and output in accepted format'
    )
    parser.add_argument('--mapping', '-m', help='Mapping file (e.g., graph_002_mapping.txt)')
    parser.add_argument('--vectors', '-v', help='Vectors input (file path or dictionary string)')
    parser.add_argument('--output', '-o', help='Output file for reordered vectors')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        mapping_file = input("Enter path to mapping file: ").strip()
        vectors_input = input("Enter vectors (file path or dictionary string): ").strip()
        output_file = input("Enter output file path: ").strip()
    else:
        # Command line mode
        if not args.mapping or not args.vectors or not args.output:
            parser.print_help()
            print("\nError: --mapping, --vectors, and --output are required (unless using --interactive)")
            sys.exit(1)
        
        mapping_file = args.mapping
        vectors_input = args.vectors
        output_file = args.output
    
    # Validate input files
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file '{mapping_file}' not found")
        sys.exit(1)
    
    # Process the vectors
    success = process_vectors_with_mapping(mapping_file, vectors_input, output_file)
    
    if success:
        print("\n✅ Processing completed successfully!")
    else:
        print("\n❌ Processing completed with errors. Check the output for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
