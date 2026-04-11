#!/bin/bash

# AICAS 2026 Submission ZIP Creator
# Creates the submission package according to UPLOAD_README.md

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT_ID="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD)"
BASE_OUTPUT_NAME="${1:-submission.zip}"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

if [[ "$BASE_OUTPUT_NAME" == *.zip ]]; then
    OUTPUT_NAME="${BASE_OUTPUT_NAME%.zip}_${COMMIT_ID}.zip"
else
    OUTPUT_NAME="${BASE_OUTPUT_NAME}_${COMMIT_ID}.zip"
fi

# Files to include
WRAPPER_FILE="evaluation_wrapper.py"
REQUIREMENTS_FILE="requirements.txt"

# Check required files
if [ ! -f "$SCRIPT_DIR/$WRAPPER_FILE" ]; then
    echo "Error: $WRAPPER_FILE not found!"
    exit 1
fi

build_cuda_extension() {
    local setup_script="$1"
    local output_pattern="$2"

    echo "Building CUDA extension via cuda_extensions/$setup_script"
    (
        cd "$SCRIPT_DIR/cuda_extensions"
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
            "$PYTHON_BIN" "$setup_script" build_ext --inplace
    )

    if ! compgen -G "$SCRIPT_DIR/cuda_extensions/$output_pattern" > /dev/null; then
        echo "Error: expected CUDA extension output matching $output_pattern not found"
        exit 1
    fi
}

build_cuda_extension "setup.py" "cuda_decode_attention_ext*.so"
build_cuda_extension "setup_decode_gemv.py" "cuda_decode_gemv_ext*.so"
build_cuda_extension "setup_decode_small_ops.py" "cuda_decode_small_ops_ext*.so"

# Create temporary directory for staging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy files to temp directory
cp "$SCRIPT_DIR/$WRAPPER_FILE" "$TEMP_DIR/"

# Include requirements.txt if it exists
if [ -f "$SCRIPT_DIR/$REQUIREMENTS_FILE" ]; then
    cp "$SCRIPT_DIR/$REQUIREMENTS_FILE" "$TEMP_DIR/"
    echo "Including $REQUIREMENTS_FILE"
fi

# Include my_kernel directory if it exists
if [ -d "$SCRIPT_DIR/my_kernel" ]; then
    cp -r "$SCRIPT_DIR/my_kernel" "$TEMP_DIR/"
    echo "Including my_kernel/ directory"
fi

# Include optimized_layer.py if it exists
if [ -f "$SCRIPT_DIR/optimized_layer.py" ]; then
    cp "$SCRIPT_DIR/optimized_layer.py" "$TEMP_DIR/"
    echo "Including optimized_layer.py"
fi

# Include CUDA Python wrappers if they exist
for py_file in cuda_decode_gemv.py cuda_decode_attention.py cuda_decode_small_ops.py; do
    if [ -f "$SCRIPT_DIR/$py_file" ]; then
        cp "$SCRIPT_DIR/$py_file" "$TEMP_DIR/"
        echo "Including $py_file"
    fi
done

# Include selected CUDA extension sources / setup files / prebuilt shared libraries
if [ -d "$SCRIPT_DIR/cuda_extensions" ]; then
    mkdir -p "$TEMP_DIR/cuda_extensions"
    for ext_file in \
        README.md \
        setup.py \
        setup_decode_gemv.py \
        setup_decode_small_ops.py \
        decode_attention_ext.cpp \
        decode_attention_kernel.cu \
        decode_gemv_ext.cpp \
        decode_gemv_kernel.cu \
        decode_gemv_gridsync.cu \
        decode_small_ops_ext.cpp \
        decode_small_ops_kernel.cu \
        decode_gemv_gridsync.o \
        decode_gemv_gridsync_dlink.o
    do
        if [ -f "$SCRIPT_DIR/cuda_extensions/$ext_file" ]; then
            cp "$SCRIPT_DIR/cuda_extensions/$ext_file" "$TEMP_DIR/cuda_extensions/"
            echo "Including cuda_extensions/$ext_file"
        fi
    done

    for so_file in "$SCRIPT_DIR"/cuda_extensions/cuda_decode_*_ext*.so; do
        if [ -f "$so_file" ]; then
            cp "$so_file" "$TEMP_DIR/cuda_extensions/"
            echo "Including cuda_extensions/$(basename "$so_file")"
        fi
    done
fi

# Include documentation (code reading guide + optimization report)
mkdir -p "$TEMP_DIR/docs"
for doc_file in code_reading_guide.md optimization_report.md; do
    if [ -f "$SCRIPT_DIR/docs/$doc_file" ]; then
        cp "$SCRIPT_DIR/docs/$doc_file" "$TEMP_DIR/docs/"
        echo "Including docs/$doc_file"
    fi
done

# Create the ZIP file
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/$OUTPUT_NAME" ./*

# Verify key runtime modules exist in the packaged tree.
python3 - "$SCRIPT_DIR/$OUTPUT_NAME" <<'PY'
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
required_files = [
    "evaluation_wrapper.py",
    "optimized_layer.py",
    "cuda_decode_attention.py",
    "cuda_decode_gemv.py",
    "cuda_decode_small_ops.py",
    "cuda_extensions/setup.py",
    "cuda_extensions/setup_decode_gemv.py",
    "cuda_extensions/setup_decode_small_ops.py",
]
with zipfile.ZipFile(zip_path) as zf:
    names = set(zf.namelist())
missing = [name for name in required_files if name not in names]
if missing:
    raise SystemExit(f"Packaged ZIP is missing required files: {missing}")

required_modules = [
    "evaluation_wrapper",
    "optimized_layer",
    "cuda_decode_attention",
    "cuda_decode_gemv",
    "cuda_decode_small_ops",
]
with tempfile.TemporaryDirectory() as td:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(td)
    sys.path.insert(0, td)
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise SystemExit(f"Packaged ZIP is missing importable modules: {missing_modules}")
PY

echo "Created: $SCRIPT_DIR/$OUTPUT_NAME"
echo "Contents:"
unzip -l "$SCRIPT_DIR/$OUTPUT_NAME"
