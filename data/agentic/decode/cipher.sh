#!/bin/bash
# Decode script: decodes a base64-encoded string passed as argument
# Usage: ./cipher.sh <base64_string>
if [ -z "$1" ]; then
    echo "Usage: ./cipher.sh <base64_string>"
    exit 1
fi
echo "$1" | base64 -d
echo ""
