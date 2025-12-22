#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
CONTAINERFILE_REL=".container/clang-format/Containerfile"
CONTAINERFILE="$REPO_ROOT/$CONTAINERFILE_REL"
IMAGE_NAME="${CLANG_FORMAT_IMAGE_NAME:-clang-format}"

# Detect container runtime
# if command -v docker &>/dev/null; then
#     OCIRUN="docker"
#     USER_FLAG="--user $(id -u):$(id -g)"
# el
if command -v podman &>/dev/null; then
    OCIRUN="podman"
    USER_FLAG="--userns=keep-id"
else
    echo "Neither Docker nor Podman is installed." >&2
    exit 1
fi

# Ensure Containerfile is tracked and get its blob hash (stable tag)
if ! git -C "$REPO_ROOT" ls-files --error-unmatch "$CONTAINERFILE_REL" >/dev/null 2>&1; then
    echo "Containerfile is not tracked by git: $CONTAINERFILE_REL" >&2
    exit 1
fi

IMAGE_VERSION="$(git -C "$REPO_ROOT" ls-tree HEAD "$CONTAINERFILE_REL" | awk '{print $3}')"

if [[ -z "$IMAGE_VERSION" ]]; then
    echo "Failed to compute git blob hash for $CONTAINERFILE_REL" >&2
    exit 1
fi

# Local image reference
LOCAL_IMAGE_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"

# Remote image reference (GHCR) used in CI
# Can be overridden explicitly:
#   CLANG_FORMAT_IMAGE=ghcr.io/owner/repo/clang-format:<tag>
# Or by overriding the repo prefix:
#   CLANG_FORMAT_GHCR_REPO=ghcr.io/owner/repo/clang-format
REMOTE_REPO="${CLANG_FORMAT_GHCR_REPO:-}"
if [[ -z "$REMOTE_REPO" && -n "${GITHUB_REPOSITORY:-}" ]]; then
    REMOTE_REPO="ghcr.io/${GITHUB_REPOSITORY,,}/clang-format"
fi
REMOTE_IMAGE_TAG=""
if [[ -n "$REMOTE_REPO" ]]; then
    REMOTE_IMAGE_TAG="${REMOTE_REPO}:${IMAGE_VERSION}"
fi

is_ci=false
if [[ "${GITHUB_ACTIONS:-}" == "true" || "${CI:-}" == "true" ]]; then
    is_ci=true
fi

ensure_local_image() {
    if ! $OCIRUN image inspect "$LOCAL_IMAGE_TAG" &>/dev/null; then
        echo "Building local image: $LOCAL_IMAGE_TAG" >&2
        $OCIRUN build -f "$CONTAINERFILE" -t "$LOCAL_IMAGE_TAG" "$REPO_ROOT"
    fi
}

pull_if_needed() {
    local img="$1"
    if ! $OCIRUN image inspect "$img" &>/dev/null; then
        echo "Pulling image: $img" >&2
        $OCIRUN pull "$img"
    fi
}

# Resolution strategy:
# 1) If CLANG_FORMAT_IMAGE is explicitly set: use it (pull if needed).
# 2) In CI: prefer pulling GHCR image (if configured), fallback to local build.
# 3) Otherwise: local build/run.
if [[ -n "${CLANG_FORMAT_IMAGE:-}" ]]; then
    IMAGE_TAG="$CLANG_FORMAT_IMAGE"
    pull_if_needed "$IMAGE_TAG"
else
    if $is_ci && [[ -n "$REMOTE_IMAGE_TAG" ]]; then
        # Prefer remote in CI
        if $OCIRUN image inspect "$REMOTE_IMAGE_TAG" &>/dev/null; then
            IMAGE_TAG="$REMOTE_IMAGE_TAG"
        else
            if $OCIRUN pull "$REMOTE_IMAGE_TAG" &>/dev/null; then
                IMAGE_TAG="$REMOTE_IMAGE_TAG"
            else
                echo "Warning: could not pull $REMOTE_IMAGE_TAG; falling back to local build." >&2
                ensure_local_image
                IMAGE_TAG="$LOCAL_IMAGE_TAG"
            fi
        fi
    else
        ensure_local_image
        IMAGE_TAG="$LOCAL_IMAGE_TAG"
    fi
fi

export REPO_ROOT OCIRUN USER_FLAG IMAGE_TAG
