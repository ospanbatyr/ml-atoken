#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

echo "Downloading AToken checkpoints..."

# Base URL
BASE_URL="https://ml-site.cdn-apple.com/models/atoken"

# Download main models
echo "Downloading AToken-So/D..."
wget -O checkpoints/atoken-sod.pt "${BASE_URL}/atoken-sod.pt"

echo "All checkpoints downloaded successfully to ./checkpoints/"
