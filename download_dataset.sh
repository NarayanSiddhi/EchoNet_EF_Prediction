#!/bin/bash

# EchoNet Pediatric Dataset Download Script
# This script helps download the full EchoNet Pediatric dataset from Azure Blob Storage
#
# Usage:
#   ./download_dataset.sh [DOWNLOAD_URL]
#   If DOWNLOAD_URL is not provided, the script will prompt for it

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/Dataset"
AZCOPY_DIR="${DATASET_DIR}/azcopy_linux_amd64"
DOWNLOAD_URL="${1:-}"

echo "=========================================="
echo "EchoNet Pediatric Dataset Download"
echo "=========================================="
echo ""

# Step 1: Check if azcopy is already installed
if command -v azcopy &> /dev/null; then
    echo "✓ azcopy is already installed"
    AZCOPY_CMD="azcopy"
else
    echo "azcopy not found. Will download it..."
    
    # Create Dataset directory
    mkdir -p "${DATASET_DIR}"
    cd "${DATASET_DIR}"
    
    # Download azcopy for Linux
    echo "Downloading azcopy for Linux..."
    AZCOPY_URL="https://aka.ms/downloadazcopy-v10-linux"
    AZCOPY_ZIP="${DATASET_DIR}/azcopy.tar.gz"
    
    if [ ! -f "${AZCOPY_ZIP}" ]; then
        wget -O "${AZCOPY_ZIP}" "${AZCOPY_URL}" || curl -L -o "${AZCOPY_ZIP}" "${AZCOPY_URL}"
    fi
    
    # Extract azcopy
    echo "Extracting azcopy..."
    tar -xzf "${AZCOPY_ZIP}" -C "${DATASET_DIR}"
    
    # Find the extracted directory (it has a version number)
    AZCOPY_EXTRACTED=$(find "${DATASET_DIR}" -maxdepth 1 -type d -name "azcopy_linux_amd64_*" | head -n 1)
    
    if [ -z "${AZCOPY_EXTRACTED}" ]; then
        echo "Error: Could not find extracted azcopy directory"
        exit 1
    fi
    
    AZCOPY_CMD="${AZCOPY_EXTRACTED}/azcopy"
    chmod +x "${AZCOPY_CMD}"
    
    echo "✓ azcopy downloaded and ready"
fi

echo ""
echo "=========================================="
echo "IMPORTANT: Get Download URL"
echo "=========================================="
echo ""
echo "To download the EchoNet Pediatric dataset, you need to:"
echo ""
echo "1. Visit: https://stanfordaimi.azurewebsites.net/datasets/a84b6be6-0d33-41f9-8996-86e5df53b005"
echo ""
echo "2. Sign in with your Stanford account (or create one if needed)"
echo ""
echo "3. Click 'Download' to get the Azure Blob Storage URL with SAS token"
echo ""
echo "4. The URL will look like:"
echo "   https://[storageaccount].blob.core.windows.net/[container]/[path]?[SAS-token]"
echo ""

# If URL not provided as argument, prompt for it
if [ -z "${DOWNLOAD_URL}" ]; then
    read -p "Paste the download URL here (or press Enter to skip and download manually later): " DOWNLOAD_URL
fi

if [ -z "${DOWNLOAD_URL}" ]; then
    echo ""
    echo "Skipping download. When you have the URL, run:"
    echo "  ${AZCOPY_CMD} copy \"<DOWNLOAD_URL>\" \"${DATASET_DIR}/EchoNet-Pediatric\" --recursive"
    echo ""
    echo "Or manually download from:"
    echo "  https://stanfordaimi.azurewebsites.net/datasets/a84b6be6-0d33-41f9-8996-86e5df53b005"
    exit 0
fi

# Step 2: Download the dataset
echo ""
echo "=========================================="
echo "Downloading Dataset"
echo "=========================================="
echo ""
echo "This may take a while (dataset is ~XX GB)..."
echo ""

TARGET_DIR="${DATASET_DIR}/EchoNet-Pediatric"
mkdir -p "${TARGET_DIR}"

# Download using azcopy
echo "Running azcopy..."
${AZCOPY_CMD} copy "${DOWNLOAD_URL}" "${TARGET_DIR}" --recursive

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Dataset downloaded to: ${TARGET_DIR}"
echo ""
echo "Next steps:"
echo "1. Verify the dataset structure contains 'pediatric_echo_avi' directory"
echo "2. Update preprocessing/config.yaml with the correct dataset_root path"
echo "3. Run preprocessing: python preprocessing/preprocess.py --config preprocessing/config.yaml"
echo ""

