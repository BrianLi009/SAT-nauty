#!/bin/bash

input_file="subgraphs_25_adj.log"
output_file="subgraphs_25_adj_rcl.log"

# Clear or create the output file
> "$output_file"

# Read the input file line by line
while IFS= read -r line; do
    # Run RCL and extract only the output string (last line, after "Output string: ")
    ../nauty2_8_8/RCL "$line" | grep "Output string:" | cut -d' ' -f3 >> "$output_file"
done < "$input_file"