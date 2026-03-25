#!/bin/bash

# AICAS 2026 Submission ZIP Creator
# Creates the submission package according to UPLOAD_README.md

set -e

# Configuration
OUTPUT_NAME="${1:-submission.zip}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Create the ZIP file
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/$OUTPUT_NAME" ./*

echo "Created: $SCRIPT_DIR/$OUTPUT_NAME"
echo "Contents:"
unzip -l "$SCRIPT_DIR/$OUTPUT_NAME"
