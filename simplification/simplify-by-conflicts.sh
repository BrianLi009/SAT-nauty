#!/bin/bash

# Default values
pseudo_check=true
lex_order="smallest"
orbit_val=""
unembeddable=true
partition_val=""
ortho=false
complex_mode=false
vectors_file=""
order=""
conflicts=10000
filename=""

# Function to display help
show_help() {
  echo "Usage: $0 <filename> [options]"
  echo "Options:"
  echo "  --order <value>      Set the order value (required)"
  echo "  --conflicts <value>  Set the number of conflicts (default: 10000)"
  echo "  --partition <value>  Set partition value"
  echo "  --ortho              Enable ortho mode"
  echo "  --complex            Enable complex mode"
  echo "  --vectors-file <file> Set vectors file"
  echo "  --lex-greatest       Use lex-greatest ordering (default: smallest)"
  echo "  --no-pseudo-check    Disable pseudo check"
  echo "  --no-unembeddable    Disable unembeddable check"
  echo "  --orbit <value>      Set orbit value"
  echo "  -h, --help           Show this help message"
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --order)
      order="$2"
      shift 2
      ;;
    --conflicts)
      conflicts="$2"
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
    --lex-greatest)
      lex_order="greatest"
      shift
      ;;
    --no-pseudo-check)
      pseudo_check=false
      shift
      ;;
    --no-unembeddable)
      unembeddable=false
      shift
      ;;
    --orbit)
      orbit_val="$2"
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

# Create necessary directories
mkdir -p log

f_dir=$filename
f_base=$(basename "$filename")

# Construct base CaDiCaL command
cmd="./cadical-rcl/build/cadical $f_dir"
cmd="$cmd --order $order --unembeddable-check 0"
[ "$pseudo_check" = false ] && cmd="$cmd --no-pseudo-check"
[ "$lex_order" = "greatest" ] && cmd="$cmd --lex-greatest"
[ -n "$orbit_val" ] && cmd="$cmd --orbit $orbit_val"
[ -n "$partition_val" ] && cmd="$cmd --partition $partition_val"
[ "$ortho" = true ] && cmd="$cmd --ortho"
[ "$complex_mode" = true ] && cmd="$cmd --complex"
[ -n "$vectors_file" ] && cmd="$cmd --vectors-file $vectors_file"
cmd="$cmd -o $f_dir.simp1 -e $f_dir.ext -n -c $conflicts"

# Print the command for debugging
echo "Executing CaDiCaL command: $cmd"

# Simplify for specified conflicts
echo "Simplifying for $conflicts conflicts"

# Execute CaDiCaL command
$cmd | tee "$f_dir".simplog

# Output final simplified instance
concat_cmd="./gen_cubes/concat-edge.sh $order \"$f_dir\".simp1 \"$f_dir\".ext > \"$f_dir\".simp"
echo "Executing concat command: $concat_cmd"
./gen_cubes/concat-edge.sh $order "$f_dir".simp1 "$f_dir".ext > "$f_dir".simp
rm -f "$f_dir".simp1
