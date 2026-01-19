import sys
import subprocess

def get_unit_clauses_from_encoding(encoding_file, index):
    """Get unit clauses by calling to_var.py with the given file and index."""
    try:
        # Call to_var.py and capture its output
        result = subprocess.run(['python3', 'to_var.py', encoding_file, str(index)], 
                              capture_output=True, text=True, check=True)
        # Split the output into individual literals and remove empty strings
        literals = result.stdout.strip().split()
        return [int(lit) for lit in literals if lit]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running to_var.py: {e}")

def update_cnf_file(input_cnf_file, unit_clauses, output_cnf_file=None):
    """Update the CNF file by adding unit clauses and incrementing the clause count.
    Memory efficient version that processes file line by line.
    """
    if output_cnf_file is None:
        output_cnf_file = input_cnf_file

    # First pass: count clauses and find number of variables
    num_vars = 0
    num_clauses = 0
    with open(input_cnf_file, 'r') as f:
        for line in f:
            if line.startswith('p cnf'):
                _, _, num_vars, orig_clauses = line.split()
                num_vars = int(num_vars)
                num_clauses = int(orig_clauses)
                break

    if num_vars == 0:
        raise ValueError("Invalid CNF file: no problem line found")

    # Second pass: write to output file
    with open(input_cnf_file, 'r') as infile, open(output_cnf_file, 'w') as outfile:
        for line in infile:
            if line.startswith('p cnf'):
                # Update the clause count in the problem line
                outfile.write(f'p cnf {num_vars} {num_clauses + len(unit_clauses)}\n')
            else:
                outfile.write(line)
        
        # Append unit clauses at the end
        for clause in unit_clauses:
            outfile.write(f'{clause} 0\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 append.py [cnf_file] [encoding_file] [index]")
        sys.exit(1)

    input_cnf_file = sys.argv[1]
    encoding_file = sys.argv[2]
    index = sys.argv[3]
    
    try:
        # Get unit clauses from to_var.py
        unit_clauses = get_unit_clauses_from_encoding(encoding_file, index)
        
        # Create output filename based on input filename and index
        output_cnf_file = f"{input_cnf_file.rsplit('.', 1)[0]}_{index}.cnf"
        
        # Update the CNF file with the unit clauses
        update_cnf_file(input_cnf_file, unit_clauses, output_cnf_file)
        print(f"Created {output_cnf_file} with unit clauses from {encoding_file} at index {index}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)