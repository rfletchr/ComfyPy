#!/usr/bin/env bash
# Bump the ComfyPy project per `uv version --bump`. Validates the argument; no default.
set -euo pipefail

# Refuse to run with uncommitted changes: a clean tree is required so the resulting
# bump commit isolates only the version change in pyproject.toml.
cd "$(git rev-parse --show-toplevel)"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "error: working tree has unstaged or staged-uncommitted changes; commit or stash first" >&2
    git status --short >&2
    exit 1
fi

usage() {
    echo "usage: $0 <major|minor|patch|stable|alpha|beta|rc|post|dev>" >&2
    exit 2
}

[ $# -eq 1 ] || usage

case "$1" in
    major|minor|patch|stable|alpha|beta|rc|post|dev) ;;
    *) usage ;;
esac

uv version --bump "$1"

# Read the new version that uv just wrote into pyproject.toml.
version="$(uv version --short)"
tag="v${version}"

# If we exit before the commit lands (abort at the prompt, error, signal),
# restore pyproject.toml so the working tree stays clean.
cleanup() {
    git checkout -- pyproject.toml 2>/dev/null || true
    echo "aborted: pyproject.toml restored to HEAD" >&2
}
trap cleanup EXIT

# Confirm a remote exists to push the tag to.
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "error: no 'origin' remote configured; cannot push tag ${tag}" >&2
    exit 1
fi

# Prompt for an annotated tag message.
echo "Tag message for ${tag} (Ctrl-C to abort):"
read -r tag_msg
if [ -z "${tag_msg}" ]; then
    echo "error: empty tag message" >&2
    exit 1
fi

# Commit the bump and create an annotated tag, then push only the tag.
GIT_EDITOR=true git commit -m "Bump version to ${version}" -- pyproject.toml
git tag -a "${tag}" -m "${tag_msg}"
git push origin "${tag}"

# Success: clear the cleanup trap so we don't revert the committed bump.
trap - EXIT