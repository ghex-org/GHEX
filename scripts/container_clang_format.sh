#!/bin/bash
set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
CONTAINERFILE="$REPO_ROOT/.container/clang-format/Containerfile"
IMAGE_NAME="clang-format"

# Detect container runtime
if command -v docker &>/dev/null; then
    OCIRUN="docker"
    USER_FLAG="--user $(id -u):$(id -g)"
elif command -v podman &>/dev/null; then
    OCIRUN="podman"
    USER_FLAG="--userns=keep-id"
else
    echo "Neither Docker nor Podman is installed." >&2
    exit 1
fi

# Get git blob hash of the Containerfile
if git ls-files --error-unmatch "$CONTAINERFILE" >/dev/null 2>&1; then
    IMAGE_VERSION=$(git ls-tree HEAD "$CONTAINERFILE" | awk '{print $3}')
else
    echo "Containerfile is not tracked by git."
    exit 1
fi

IMAGE_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"

# Build the image if not present
if ! $OCIRUN image inspect "$IMAGE_TAG" &>/dev/null; then
    echo "Building $IMAGE_TAG image..."
    $OCIRUN build -f "$CONTAINERFILE" -t "$IMAGE_TAG" .
fi

export OCIRUN USER_FLAG IMAGE_TAG
