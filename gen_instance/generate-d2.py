#!/usr/bin/python
import sys, getopt
from squarefree import squarefree
from triangle import triangle
from mindegree import mindegree
from noncolorable import noncolorable
from cubic import cubic
from cubic_lex_greatest import cubic as cubic_lex_greatest
from b_b_card import generate_edge_clauses
from diameter import distance_constraint
import subprocess
import os
import argparse

def generate(n, block, lower_bound=None, upper_bound=None, lex_option="lex-least"):
    """
    n: size of the graph
    block: color ratio to block
    lower_bound: lower bound for triangle count
    upper_bound: upper bound for triangle count
    lex_option: "lex-least", "lex-greatest", or "no-lex" to control isomorphism blocking
    Given n, the function calls each individual constraint-generating function, then write them into a DIMACS file as output
    The variables are listed in the following order:
    edges - n choose 2 variables
    triangles - n choose 3 variables
    extra variables from cubic
    """
    base_name = f"constraints_{n}_{block}"
    if lower_bound and upper_bound:
        base_name += f"_{lower_bound}_{upper_bound}"
    
    if lex_option == "lex-greatest":
        cnf_file = base_name + "_lex_greatest"
    elif lex_option == "no-lex":
        cnf_file = base_name + "_no_lex"
    else:  # lex-least is the default
        cnf_file = base_name + "_1"
    
    if os.path.exists(cnf_file):
        print(f"File '{cnf_file}' already exists. Terminating...")
        sys.exit()
    edge_dict = {}
    tri_dict = {}
    count = 0
    clause_count = 0
    
    # Generate edge variables
    edge_start = 1
    for j in range(1, n+1):
        for i in range(1, j):
            count += 1
            edge_dict[(i,j)] = count
    edge_end = count
    
    # Generate triangle variables
    tri_start = edge_end + 1
    for a in range(1, n-1):
        for b in range(a+1, n):
            for c in range(b+1, n+1):
                count += 1
                tri_dict[(a,b,c)] = count
    tri_end = count
    
    # Print variable ranges
    print("\n=== VARIABLE RANGES ===")
    print(f"Edge variables: {edge_start} to {edge_end} ({edge_end - edge_start + 1} variables)")
    print(f"Triangle variables: {tri_start} to {tri_end} ({tri_end - tri_start + 1} variables)")
    
    # Apply constraints
    clause_count += squarefree(n, edge_dict, cnf_file)
    print("\n=== CONSTRAINTS ===")
    print("Graph is squarefree")
    
    clause_count += triangle(n, edge_dict, tri_dict, cnf_file)
    print("All vertices are part of a triangle")
    
    clause_count += noncolorable(n, edge_dict, tri_dict, cnf_file, block)
    print("Graph is noncolorable")
    
    clause_count += mindegree(n, 3, edge_dict, cnf_file)
    print("Minimum degree of each vertex is 3")
    
    # Add the distance constraint - each vertex in B has distance at most 2 to at least one vertex in A
    # We need to track the maximum variable ID used by distance_constraint
    distance_start = tri_end + 1
    new_clauses, new_var_count = distance_constraint(n, edge_dict, cnf_file, A_size=13, d=2)
    clause_count += new_clauses
    distance_end = new_var_count
    count = new_var_count  # Update the variable count for cubic
    
    print("Distance constraint applied: vertices beyond 13 have distance ≤ 2 to at least one vertex in first 13")
    print(f"Distance constraint variables: {distance_start} to {distance_end} ({distance_end - distance_start + 1} variables)")
    
    """
    conway(n, edge_dict, tri_dict, 1, 3)
    conway(n, edge_dict, tri_dict, 2, 4)
    conway(n, edge_dict, tri_dict, 3, 4)
    print ("conway constraint")
    """
    # Apply isomorphism blocking only if not using no-lex
    if lex_option != "no-lex":
        # Choose which cubic implementation to use
        cubic_impl = cubic_lex_greatest if lex_option == "lex-greatest" else cubic
        print(f"Using {lex_option} ordering for isomorphism blocking (but not applying it)")
        
        # Skip isomorphism blocking but keep the variable count
        var_count = count
        print("Isomorphism blocking commented out")
    else:
        print("Skipping isomorphism blocking (no-lex option)")
        var_count = count  # No additional variables added
    
    # Add triangle count constraints if bounds are provided
    if lower_bound and upper_bound:
        tri_vars = [v for k, v in sorted(tri_dict.items())]
        card_start = var_count + 1
        var_count_card, clause_count_card = generate_edge_clauses(tri_vars, int(lower_bound), int(upper_bound), var_count, cnf_file)
        card_end = var_count_card
        var_count = var_count_card
        clause_count += clause_count_card
        
        print("Triangle count constraints applied")
        print(f"Triangle count constraint variables: {card_start} to {card_end} ({card_end - card_start + 1} variables)")
    
    # Print final summary
    print("\n=== SUMMARY ===")
    print(f"Total variables: {var_count}")
    print(f"Total clauses: {clause_count}")
    print(f"Variable breakdown:")
    print(f"  - Edge variables: {edge_start}-{edge_end} ({edge_end - edge_start + 1} variables)")
    print(f"  - Triangle variables: {tri_start}-{tri_end} ({tri_end - tri_start + 1} variables)")
    print(f"  - Distance constraint variables: {distance_start}-{distance_end} ({distance_end - distance_start + 1} variables)")
    
    if lower_bound and upper_bound:
        print(f"  - Triangle count constraint variables: {card_start}-{card_end} ({card_end - card_start + 1} variables)")
    
    firstline = 'p cnf ' + str(var_count) + ' ' + str(clause_count)
    subprocess.call(["./gen_instance/append.sh", cnf_file, cnf_file+"_new", firstline])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate constraints for graph problems.')
    parser.add_argument('n', type=int, help='Size of the graph')
    parser.add_argument('block', type=str, help='Block identifier')
    parser.add_argument('lex_option', type=str, nargs='?', default="lex-least", 
                      help='Isomorphism blocking option: "lex-least" (default), "lex-greatest", or "no-lex"')
    parser.add_argument('--lower', type=int, help='Lower bound for triangle count')
    parser.add_argument('--upper', type=int, help='Upper bound for triangle count')
    
    args = parser.parse_args()
    
    # Validate lex_option
    if args.lex_option not in ["lex-least", "lex-greatest", "no-lex"]:
        parser.error("lex_option must be one of: lex-least, lex-greatest, no-lex")
    
    # Validate bounds
    if (args.lower is None) != (args.upper is None):
        parser.error("Both lower and upper bounds must be provided together or omitted")
    
    generate(args.n, args.block, args.lower, args.upper, args.lex_option)
