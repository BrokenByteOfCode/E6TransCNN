import argparse
import sys
from pathlib import Path
from model.utils.dataset import build_vocabularies

def main():
    parser = argparse.ArgumentParser(description="E6TransCNN Vocabulary Generator")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="raw",
        help="Path to the directory containing raw JSON and image files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vocab",
        help="Directory where the generated vocabularies.json will be saved"
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=5,
        help="Minimum frequency threshold for a tag to be included in the vocabulary"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    
    if not data_path.exists() or not data_path.is_dir():
        print(f"Error: Input data directory '{data_path}' does not exist or is not a directory.")
        sys.exit(1)
        
    build_vocabularies(
        data_dir=data_path,
        output_dir=output_path,
        min_freq=args.min_freq
    )
    
    final_file = output_path / "vocabularies.json"
    print(f"Success: Vocabulary generation complete. File saved at: {final_file.resolve()}")

if __name__ == "__main__":
    main()