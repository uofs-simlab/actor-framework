import os
import sys
import json
import struct
import shutil
import numpy as np
import ssgetpy

from scipy.io import mmread
from scipy.sparse import issparse

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "valid_matrix_subset.txt"
OUTPUT_DIR = "./downloaded_matrices"

MATRIX_DIR = os.path.join(OUTPUT_DIR, "matrices", "spd")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata", "spd")
INDEX_PATH = os.path.join(OUTPUT_DIR, "spd_index.json")

# ============================================================
# SETUP & PARSING
# ============================================================
def setup_directories():
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

def parse_matrix_names(file_path):
    """Extracts matrix names from the file paths (e.g., '1138_bus' from '.../1138_bus.bin')"""
    if not os.path.exists(file_path):
        print(f"[ERROR] Input file {file_path} not found.")
        sys.exit(1)
        
    matrix_names = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # Get the filename (e.g., 1138_bus.bin) and strip extension
                base = os.path.basename(line)
                name, _ = os.path.splitext(base)
                matrix_names.append(name)
    return matrix_names

# ============================================================
# UTILITIES
# ============================================================
def find_mtx(meta_name):
    for root, dirs, files in os.walk(os.path.expanduser("~/.ssgetpy")):
        for file in files:
            if file.lower() == f"{meta_name.lower()}.mtx":
                return os.path.join(root, file)
    return None

def export_binary(csr, output_path):
    coo = csr.tocoo()
    rows = int(coo.shape[0])
    cols = int(coo.shape[1])
    nnz = int(coo.nnz)

    row_indices = coo.row.astype(np.int32)
    col_indices = coo.col.astype(np.int32)
    values = coo.data.astype(np.float32)

    with open(output_path, "wb") as f:
        f.write(struct.pack("iii", rows, cols, nnz))
        f.write(row_indices.tobytes())
        f.write(col_indices.tobytes())
        f.write(values.tobytes())

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    setup_directories()
    matrix_names = parse_matrix_names(INPUT_FILE)
    
    print(f"[INFO] Found {len(matrix_names)} matrices to process from {INPUT_FILE}")
    index_entries = []

    for idx, name in enumerate(matrix_names, 1):
        print(f"\n[INFO] Processing [{idx}/{len(matrix_names)}]: {name}")
        
        # Search for exact name match in SuiteSparse
        results = ssgetpy.search(name=name)
        if not results:
            print(f"[WARN] Matrix '{name}' not found in SuiteSparse Collection. Skipping.")
            continue
            
        # Grab the first match (ssgetpy ranks exact matches highly)
        meta = results[0]
        
        try:
            matrices = ssgetpy.fetch(meta.id)
            if matrices:
                matrices.download(extract=True)
            else:
                print(f"[SKIP] Fetch failed for {meta.name}")
                continue
        except Exception as e:
            print(f"[WARN] Download failed for {meta.name}: {e}")
            continue

        # Locate downloaded .mtx file
        mtx_file = find_mtx(meta.name)
        if mtx_file is None:
            print(f"[SKIP] no .mtx file found locally for: {meta.name}")
            continue

        # Process and write file
        try:
            mat = mmread(mtx_file)
            if not issparse(mat):
                print(f"[SKIP] Not a sparse matrix: {meta.name}")
                continue

            csr = mat.tocsr()
            
            # Paths configuration
            bin_path = os.path.join(MATRIX_DIR, f"{name}.bin")
            meta_path = os.path.join(METADATA_DIR, f"{name}.json")

            # Export custom binary format
            export_binary(csr, bin_path)

            # Build metadata object
            metadata = {
                "id": meta.id,
                "name": meta.name,
                "group": meta.group,
                "rows": int(csr.shape[0]),
                "cols": int(csr.shape[1]),
                "nnz": int(csr.nnz),
                "paths": {
                    "binary": bin_path
                }
            }

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            index_entries.append({
                "name": meta.name,
                "binary": bin_path,
                "metadata": meta_path
            })

            print(f"[PASS] Successfully processed and saved {meta.name}")

        except Exception as e:
            print(f"[WARN] Failed processing data for {meta.name}: {e}")

        # Clean cache to prevent disk explosion
        shutil.rmtree(os.path.expanduser("~/.ssgetpy"), ignore_errors=True)

    # Save the execution index mapping
    with open(INDEX_PATH, "w") as f:
        json.dump(index_entries, f, indent=2)

    print(f"\n[DONE] Pipeline complete. Saved execution index map to {INDEX_PATH}")