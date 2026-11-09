"""Upload a local folder to the Hugging Face Hub.

The token is read from the HF_TOKEN environment variable (or the cached
`hf auth login` credentials) -- never hardcode it.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/upload.py --folder /workspace --repo-id ManhQuangAI/workspace-deepFake
"""

import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default="/workspace", help="Local folder to upload")
    parser.add_argument("--repo-id", default="ManhQuangAI/workspace-deepFake")
    parser.add_argument("--repo-type", default="dataset", choices=["model", "dataset", "space"])
    parser.add_argument("--private", action="store_true", help="Create the repo as private (default: public)")
    args = parser.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # None -> falls back to cached login

    print(f"Uploading {args.folder} to {args.repo_id} ({args.repo_type})...")
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.folder, repo_id=args.repo_id, repo_type=args.repo_type)
    print("Upload complete.")


if __name__ == "__main__":
    main()
