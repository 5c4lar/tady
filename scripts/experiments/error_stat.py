import pathlib
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Dir contains the error files")
    args = parser.parse_args()

    root_dir = pathlib.Path(args.input)
    files = list(root_dir.rglob("*.json"))  # Convert generator to list
    errors = [(file, json.load(open(file, "r"))) for file in files]
    total = len(errors)
    error_files = 0
    error_by_category = {"coexist": 0, "dangling": 0, "exclusive": 0}
    for file, error in errors:
        if error["coexist"] or error["dangling"] or error["exclusive"]:
            error_files += 1
            for key in error:
                if error[key]:
                    # print(f"{file}: {key}")
                    error_by_category[key] += 1
                    if key == "dangling":
                        print(f"{file}: {key}")
    for category, count in error_by_category.items():
        print(f"{category}: {count}, {count / total:.4f}")
    print(f"Total files: {total}")
    print(f"Error files: {error_files}")
    print(f"Error rate: {error_files / total:.4f}")

if __name__ == "__main__":
    main()