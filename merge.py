# /// script
# dependencies = [
#   "python-dotenv",
#   "huggingface-hub",
#   "hf-transfer",
#   "tqdm",
# ]
# ///

import os

# ── Speed flags (MUST be set BEFORE importing huggingface_hub) ──
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"      # Multi-part parallel chunk downloads
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"          # Xet high-perf transfer layer
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"        # 5 min timeout per request

import sys
import subprocess
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
HF_OUTPUT_REPO: str = "anisoleai/embeddings"
PERCENTILE: float = 0.50  # Cutoff percentile: drops bottom 50% of counts (median threshold)
MAX_DOWNLOAD_WORKERS: int = 4  # Concurrency for batch download (4 workers to avoid API rate limits)
BATCH_SIZE: int = 8  # Number of files to process per batch (keeps disk usage under 25 GB)

# Load environment variables (checking CWD, script directory, and parent directories to find .env)
load_dotenv()
if not os.getenv("HF_TOKEN"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
if not os.getenv("HF_TOKEN"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
if not os.getenv("HF_TOKEN"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

HF_TOKEN: str = os.getenv("HF_TOKEN", "")

def compile_executable() -> str:
    """
    Compile comat.cpp to a standalone executable. Returns path to the compiled binary.
    """
    dir_path: str = os.path.dirname(__file__)
    exe_name: str = "comat.exe" if sys.platform.startswith("win") else "comat"
    exe_path: str = os.path.join(dir_path, exe_name)
    cpp_path: str = os.path.join(dir_path, "comat.cpp")

    print(f"Compiling C++ core executable {cpp_path} -> {exe_path}...", flush=True)
    cmd: List[str] = ["g++", "-O3", "-std=gnu++11", "-march=native", "-o", exe_path, cpp_path]
    subprocess.run(cmd, check=True)
    print("C++ Core compiled successfully.", flush=True)
    return exe_path

def download_single(remote_path: str, local_dir: str) -> str:
    """
    Download a single file from HF using hf_hub_download with cache-bypass disk optimization.
    Returns the local path of the downloaded file.
    """
    return hf_hub_download(
        repo_id=HF_OUTPUT_REPO,
        filename=remote_path,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set. Please add it to your environment.", file=sys.stderr, flush=True)
        sys.exit(1)

    print("============================================================", flush=True)
    print("Batching Parallel PPMI & Co-occurrence Merger", flush=True)
    print("============================================================", flush=True)

    # 1. Compile C++ core executable
    exe_path: str = compile_executable()

    # 2. List remote counts files
    api: HfApi = HfApi(token=HF_TOKEN)
    print(f"Listing dataset files in '{HF_OUTPUT_REPO}'...", flush=True)
    all_files: List[str] = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset")

    counts_files: List[str] = []
    for f in all_files:
        parts: List[str] = f.split('/')
        if len(parts) == 2 and parts[0].startswith("data_") and parts[1].startswith("counts_") and parts[1].endswith(".bin"):
            counts_files.append(f)

    counts_files.sort()
    num_files: int = len(counts_files)
    print(f"Found {num_files} counts files to merge.", flush=True)

    if num_files == 0:
        print("No counts files found to merge.", flush=True)
        sys.exit(0)

    # Setup directories
    dir_path: str = os.path.dirname(__file__)
    local_cache_dir: str = os.path.join(dir_path, "data_cache")
    os.makedirs(local_cache_dir, exist_ok=True)

    # Incremental Merging Loop
    temp_merged_path: str = ""
    num_batches: int = (num_files + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(num_batches):
        start_idx: int = b_idx * BATCH_SIZE
        end_idx: int = min(start_idx + BATCH_SIZE, num_files)
        batch_files: List[str] = counts_files[start_idx:end_idx]
        is_last: bool = (end_idx >= num_files)

        print(f"\n==========================================", flush=True)
        print(f"Processing Batch {b_idx + 1}/{num_batches} (Shards {start_idx} to {end_idx - 1})", flush=True)
        print(f"==========================================", flush=True)

        # 3. Download batch shards in parallel
        print(f"Downloading {len(batch_files)} shards with {MAX_DOWNLOAD_WORKERS} workers...", flush=True)
        local_batch_paths: List[str] = [""] * len(batch_files)

        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            future_to_idx = {
                executor.submit(download_single, remote_path, local_cache_dir): idx
                for idx, remote_path in enumerate(batch_files)
            }
            with tqdm(total=len(batch_files), desc=f"Downloading batch {b_idx + 1}", unit="file") as pbar:
                for future in as_completed(future_to_idx):
                    idx: int = future_to_idx[future]
                    try:
                        local_batch_paths[idx] = future.result()
                    except Exception as e:
                        print(f"\nERROR downloading {batch_files[idx]}: {e}", flush=True)
                        sys.exit(1)
                    pbar.update(1)

        # 4. Prepare file list for C++ merger
        files_to_merge: List[str] = []
        if temp_merged_path and os.path.exists(temp_merged_path):
            files_to_merge.append(temp_merged_path)
        files_to_merge.extend(local_batch_paths)

        files_list_path: str = os.path.join(dir_path, f"files_list_{b_idx}.txt")
        with open(files_list_path, "w", encoding="utf-8") as f_list:
            for path in files_to_merge:
                f_list.write(path.replace("\\", "/") + "\n")

        # Define outputs and run mode
        if is_last:
            mode: str = "ppmi"
            output_path: str = os.path.join(dir_path, "final_ppmi_scores.csv")
        else:
            mode: str = "intermediate"
            output_path: str = os.path.join(dir_path, f"temp_merged_{b_idx}.bin")

        print(f"Launching C++ merge (Mode: {mode})...", flush=True)
        proc = subprocess.Popen([
            exe_path,
            files_list_path,
            mode,
            str(PERCENTILE),
            output_path
        ])
        proc.wait()
        if proc.returncode != 0:
            print(f"ERROR: C++ merger process exited with code {proc.returncode}.", file=sys.stderr, flush=True)
            sys.exit(proc.returncode)

        # 5. Cleanup
        if os.path.exists(files_list_path):
            os.remove(files_list_path)

        # Delete downloaded raw shards for this batch (saves disk space!)
        for path in local_batch_paths:
            if os.path.exists(path):
                os.remove(path)

        # Delete previous intermediate merged file (saves disk space!)
        if temp_merged_path and os.path.exists(temp_merged_path):
            os.remove(temp_merged_path)

        # Setup paths for next iteration
        if not is_last:
            temp_merged_path = output_path
            print(f"Batch {b_idx + 1} complete. Intermediate file: {temp_merged_path} ({os.path.getsize(temp_merged_path)/1024/1024:.2f} MB)", flush=True)

    # 6. Final verification details
    final_output: str = os.path.join(dir_path, "final_ppmi_scores.csv")
    if os.path.exists(final_output):
        size_mb: float = os.path.getsize(final_output) / 1024 / 1024
        print(f"\n[Success] Final PPMI CSV saved to {final_output} ({size_mb:.2f} MB)", flush=True)
    else:
        print("ERROR: Failed to save final CSV file.", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
