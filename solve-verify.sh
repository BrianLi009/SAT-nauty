#!/bin/bash

# Default values
partition_val=""
ortho=false
complex_mode=false
vectors_file=""

# Function to display help
show_help() {
	echo "
Description:
    Script for solving CNF instances using CaDiCaL

Usage:
    ./solve-verify.sh [options] <filename>

Required Arguments:
    <filename>: file name of the CNF instance to be solved

Options:
    --order <value>      Set the order value (required)
    --partition <value>  Set partition value
    --ortho              Enable ortho mode
    --complex            Enable complex mode
    --vectors-file <file> Set vectors file
    -h, --help           Show this help message

Example:
    ./solve-verify.sh --order 32 --partition 25 --complex --vectors-file vectors.txt constraints_32_g2.cnf
"
	exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		--order)
			order="$2"
			shift 2
			;;
		--partition)
			partition_val="$2"
			shift 2
			;;
		--ortho)
			ortho=true
			shift
			;;
		--complex)
			complex_mode=true
			shift
			;;
		--vectors-file)
			vectors_file="$2"
			shift 2
			;;
		-h|--help)
			show_help
			;;
		-*)
			echo "Unknown option: $1" >&2
			show_help
			;;
		*)
			if [ -z "$filename" ]; then
				filename="$1"
				shift
			else
				echo "Unexpected argument: $1" >&2
				show_help
			fi
			;;
	esac
done

# Validate required parameters
if [ -z "$filename" ]; then
	echo "Error: Filename is required" >&2
	show_help
fi

if [ -z "$order" ]; then
	echo "Error: Order is required (use --order)" >&2
	show_help
fi

# Construct solver command based on options
cmd="./cadical-rcl/build/cadical $filename"
cmd="$cmd --order $order --unembeddable-check 0"
[ -n "$partition_val" ] && cmd="$cmd --partition $partition_val"
[ "$ortho" = true ] && cmd="$cmd --ortho"
[ "$complex_mode" = true ] && cmd="$cmd --complex"
[ -n "$vectors_file" ] && cmd="$cmd --vectors-file $vectors_file"

# Print the command
echo "Executing command: $cmd"

# Execute solver command
$cmd | tee $filename.log

# Check if UNSAT was found
if grep -q "UNSAT" "$filename.log"; then
	echo "Instance solved: UNSAT"
elif grep -q "SAT" "$filename.log"; then
	echo "Instance solved: SAT"
else
	echo "Instance not solved (UNKNOWN)"
fi
