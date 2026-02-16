import os

DOWNLOADS_DIR = "downloads"
MERGED_PREFIX = "merged_"
MAX_FILES = 20
MAX_SIZE = 512 * 1024 * 1024  # 512 MB


def get_txt_files():
    files = [f for f in os.listdir(DOWNLOADS_DIR) if f.endswith(".txt")]
    files.sort()
    return files


def merge_files():
    txt_files = get_txt_files()
    total_files = len(txt_files)
    files_per_merged = max(1, (total_files + MAX_FILES - 1) // MAX_FILES)
    merged_idx = 1
    merged_dir = "merged"
    os.makedirs(merged_dir, exist_ok=True)
    merged_path = os.path.join(merged_dir, f"merged_{merged_idx:02d}.txt")
    merged_file = open(merged_path, "w", encoding="utf-8")
    merged_size = 0
    count_in_merged = 0

    for i, fname in enumerate(txt_files):
        fpath = os.path.join(DOWNLOADS_DIR, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        header = f"\n\n===== BOOK: {fname} =====\n\n"
        chunk = header + content + "\n"
        chunk_bytes = chunk.encode("utf-8")
        if merged_size + len(chunk_bytes) > MAX_SIZE or count_in_merged >= files_per_merged:
            merged_file.close()
            merged_idx += 1
            if merged_idx > MAX_FILES:
                print("Warning: More than 20 merged files would be needed. Some files may be skipped.")
                break
            merged_path = os.path.join(merged_dir, f"merged_{merged_idx:02d}.txt")
            merged_file = open(merged_path, "w", encoding="utf-8")
            merged_size = 0
            count_in_merged = 0
        merged_file.write(chunk)
        merged_size += len(chunk_bytes)
        count_in_merged += 1
    merged_file.close()
    print(f"Merged {total_files} files into {merged_idx} files in '{merged_dir}/'.")


if __name__ == "__main__":
    merge_files()
