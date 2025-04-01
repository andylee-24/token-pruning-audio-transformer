import os
import os
import re
import argparse

def extract_score_from_file(file_path):
    """Extracts the score from the file name."""
    match = re.search(r'best-\d{3}-(\d{2,3}\.\d{4})\.txt', file_path)
    if match:
        return float(match.group(1))
    return None


def extract_score_from_file_drop_cx(file_path):
    """Extracts the score from the file name."""
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        try:
            score = float(first_line)
        except ValueError:
            score = None  # Handle the case where conversion fails
            print(f'Error: Could not convert "{first_line}" to a float in file "{file_path}"')
            assert False
    return score


def calculate_average_score(root_dir, pattern):
    """Calculates the average score from all files matching the pattern recursively."""
    total_score = 0
    count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file_name in filenames:
            if file_name.startswith(pattern) and file_name.endswith(".txt"):
                file_path = os.path.join(dirpath, file_name)
                score = extract_score_from_file_drop_cx(file_path)
                if score is not None:
                    total_score += score
                    count += 1

    if count == 0:
        return None

    return total_score / count

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score over all folds in one or more directories.")
    parser.add_argument("root_directories", type=str, nargs='+', help="Root directories containing the fold directories")
    parser.add_argument("--pattern", type=str, default='best-', help="Pattern to match the score files")
    args = parser.parse_args()

    for root_directory in args.root_directories:
        print(f"Processing directory: {root_directory}")
        average_score = calculate_average_score(root_directory, pattern=args.pattern)
        if average_score is not None:
            print(f"Average score over all folds in '{root_directory}': {average_score:.4f}")
            # Write the average score to a file with the score in the title
            score_filename = os.path.join(root_directory, f"{args.pattern}-{average_score:.4f}.txt")
            with open(score_filename, "w") as score_file:
                score_file.write(f"{average_score:.4f}\n")
        else:
            print(f"No scores found in the specified directory: {root_directory}")

if __name__ == "__main__":
    main()
