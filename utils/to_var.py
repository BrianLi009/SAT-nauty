def process_binary_string(binary_string):
    # Convert the string to positive and negative numbers
    result = []
    for i, char in enumerate(binary_string, start=1):
        result.append(i if char == "1" else -i)
    
    # Create a string for all numbers (positive and negative)
    all_numbers = " ".join(map(str, result))
    return all_numbers

def main(file_path, index):
    try:
        # Read the file and get the string at specified index
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if index >= len(lines):
                raise IndexError(f"Index {index} is out of range. File has {len(lines)} lines.")
            input_string = lines[index].strip()
        
        # Process the binary string and print results
        result = process_binary_string(input_string)
        print(result)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except IndexError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python to_var.py <file_path> <index>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    index = int(sys.argv[2])
    main(file_path, index)