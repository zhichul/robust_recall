#!/bin/bash

# Check if directory argument is supplied
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

TARGET_DIR="$1"

# Find and delete matching files under the specified directory
find "$TARGET_DIR" -type f -name 'optim_world_size_*_rank_*.pt' -exec rm -v {} \;