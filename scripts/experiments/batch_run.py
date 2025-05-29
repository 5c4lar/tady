import subprocess

def run_infer(file_path, model, output_path):
    cmd = [
        "python",
        "-m",
        "tady.infer",
        "--path", 
        str(file_path),
        "--model",
        model,
        "--output_path",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def main():
    import argparse
    import pathlib
    import json
    parser = argparse.ArgumentParser(description="Batch run script for experiments.")
    parser.add_argument("--files", type=str, required=True, help="Path to the file containing the list of files to process.")
    parser.add_argument("--dir", type=str, required=True, help="Directory where the files are located.")
    parser.add_argument("--model", type=str, required=True, help="Model to use for processing.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    args = parser.parse_args()
    base_dir = pathlib.Path(args.dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = json.load(open(args.files, "r"))
    for sample in samples:
        rel_path = pathlib.Path(sample).relative_to(base_dir)
        output_path = output_dir / f"{rel_path}.npz"
        print(f"Processing {sample} with model {args.model}...")
        run_infer(sample, args.model, output_path)
        print(f"Output saved to {output_path}")
        
if __name__ == "__main__":
    main()