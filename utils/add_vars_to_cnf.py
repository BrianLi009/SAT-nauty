def add_vars_to_cnf(input_cnf_file, vars_file, output_cnf_file=None):
    """
    Add variables from vars_file as unit clauses to a CNF formula.
    
    Parameters:
        input_cnf_file (str): Path to the input CNF file
        vars_file (str): Path to the file containing variables (like graph_25_vars.log)
        output_cnf_file (str): Path to the output CNF file. If None, overwrites input_cnf_file
    """
    # If no output file specified, overwrite input file
    if output_cnf_file is None:
        output_cnf_file = input_cnf_file
    
    # Read variables from vars file
    with open(vars_file, 'r') as f:
        vars_line = f.read().strip()
        unit_clauses = [int(var) for var in vars_line.split()]
    
    # Read the original CNF file
    with open(input_cnf_file, 'r') as f:
        cnf_lines = f.readlines()
    
    # Parse header
    header = cnf_lines[0].strip().split()
    num_vars = int(header[2])
    num_clauses = int(header[3])
    
    # Add unit clauses
    new_clauses = [f"{var} 0\n" for var in unit_clauses]
    
    # Update header with new clause count
    new_header = f"{header[0]} {header[1]} {num_vars} {num_clauses + len(unit_clauses)}\n"
    
    # Write to output file
    with open(output_cnf_file, 'w') as f:
        f.write(new_header)
        f.writelines(cnf_lines[1:])  # Write original clauses
        f.writelines(new_clauses)    # Write new unit clauses

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python add_vars_to_cnf.py <input_cnf_file> <vars_file> [output_cnf_file]")
        sys.exit(1)
    
    input_cnf = sys.argv[1]
    vars_file = sys.argv[2]
    output_cnf = sys.argv[3] if len(sys.argv) == 4 else None
    
    add_vars_to_cnf(input_cnf, vars_file, output_cnf)