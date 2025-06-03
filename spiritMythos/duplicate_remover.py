# duplicate_remover.py
"""
Utility script to remove duplicate lines (e.g., URLs) from a text file.
Usage:
    python duplicate_remover.py input.txt output.txt
If output.txt is omitted, input.txt will be overwritten with duplicates removed.
"""
import sys
import base64


def remove_duplicates(input_path, output_path=None):
    seen = set()
    unique_lines = []
    total = 0
    skipped = 0
    try:
        with open(input_path, 'rb') as infile:
            for line in infile:
                total += 1
                if total % 1000 == 0:
                    print(f"Processed {total} lines...")
                try:
                    line_clean = line.rstrip(b'\n').decode('utf-8')
                except UnicodeDecodeError:
                    # If cannot decode, base64 encode and mark
                    line_clean = 'BASE64:' + base64.b64encode(line.rstrip(b'\n')).decode('ascii')
                if line_clean and line_clean not in seen:
                    seen.add(line_clean)
                    unique_lines.append(line_clean)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    if not output_path:
        output_path = input_path
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in unique_lines:
                outfile.write(line + '\n')
        print(f"Removed duplicates. {len(unique_lines)} unique lines written to {output_path}.")
        print(f"Total lines processed: {total}. Lines base64-encoded due to decode errors: {sum(1 for l in unique_lines if l.startswith('BASE64:'))}.")
    except Exception as e:
        print(f"Error writing file: {e}")


def prompt_for_paths():
    input_path = input("Enter the path to the input file: ").strip()
    output_path = input("Enter the path to the output file (leave blank to overwrite input): ").strip()
    if not output_path:
        output_path = None
    return input_path, output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No command-line arguments provided. Switching to interactive mode.")
        input_file, output_file = prompt_for_paths()
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
    remove_duplicates(input_file, output_file)
