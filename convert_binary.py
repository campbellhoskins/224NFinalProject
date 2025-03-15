import os

# File paths
input_files = [
    'predictions/para-dev-output.csv',
    'predictions/para-test-output.csv'
]
output_files = [
    'predictions/para-dev-output-converted.txt',
    'predictions/para-test-output-converted.txt'
]

# Mapping for conversion
token_map = {
    '1': '8505',  # Token ID for 'yes'
    '0': '3919'   # Token ID for 'no'
}

def convert_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        next(infile)  # Skip the header
        for line in infile:
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue  # Skip malformed lines
            identifier, predicted = parts[0].strip(), parts[1].strip()
            token = token_map.get(predicted)
            if token:
                outfile.write(f"{identifier},{token}\n")

if __name__ == "__main__":
    for input_file, output_file in zip(input_files, output_files):
        convert_file(input_file, output_file)
        print(f"Converted {input_file} to {output_file}")
