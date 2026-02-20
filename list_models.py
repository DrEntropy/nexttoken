"""List HuggingFace models available in the local cache."""

from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()

if not cache_info.repos:
    print("No models found in the HuggingFace cache.")
    print("Run 'uv run demo.py' to download the default model.")
else:
    print("Available models (use with NEXTTOKEN_MODEL):\n")
    for repo in sorted(cache_info.repos, key=lambda r: r.repo_id):
        if repo.repo_type == "model":
            print(f"  NEXTTOKEN_MODEL={repo.repo_id}")
