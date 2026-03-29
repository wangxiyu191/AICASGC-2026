#!/bin/bash

# AICAS 2026 Submission ZIP Creator
# Creates the submission package according to UPLOAD_README.md

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT_ID="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD)"
BASE_OUTPUT_NAME="${1:-submission.zip}"

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
for py_file in cuda_decode_gemv.py cuda_decode_attention.py; do
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
        decode_attention_ext.cpp \
        decode_attention_kernel.cu \
        decode_gemv_ext.cpp \
        decode_gemv_kernel.cu
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

# Create the ZIP file
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/$OUTPUT_NAME" ./*

echo "Created: $SCRIPT_DIR/$OUTPUT_NAME"
echo "Contents:"
unzip -l "$SCRIPT_DIR/$OUTPUT_NAME"
