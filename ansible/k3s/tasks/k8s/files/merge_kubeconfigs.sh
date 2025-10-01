#!/bin/bash
set -eo pipefail

# Script to merge kubeconfig files, replacing contexts with same names
# Usage: merge_kubeconfigs.sh <staging_dir> <username>

STAGING_DIR="${1:-}"
USER="${2:-}"

# Validate inputs
if [ -z "$STAGING_DIR" ] || [ -z "$USER" ]; then
    echo "Usage: $0 <staging_dir> <username>"
    exit 1
fi

if [ ! -d "$STAGING_DIR" ]; then
    echo "Error: Staging directory $STAGING_DIR does not exist"
    exit 1
fi

# Get the user's home directory
if [ "$USER" = "root" ]; then
    USER_HOME="/root"
else
    USER_HOME="/home/${USER}"
fi

# Test kubectl availability
if ! command -v kubectl >/dev/null 2>&1; then
    echo "Error: kubectl not found in PATH"
    exit 1
fi

# Collect new config files
new_configs=()
shopt -s nullglob
for config_file in "${STAGING_DIR}"/*.yaml; do
    if [ -f "$config_file" ] && [[ "$config_file" != *"config.bak.yaml" ]] && [[ "$config_file" != *"merged-"* ]]; then
        new_configs+=("$config_file")
    fi
done
shopt -u nullglob

if [ ${#new_configs[@]} -eq 0 ]; then
    echo "No kubeconfig files found to merge for ${USER}"
    exit 1
fi

echo "Processing ${#new_configs[@]} kubeconfig(s)"

# Extract context names from new configs to identify what needs replacing
contexts_to_replace=()
for config_file in "${new_configs[@]}"; do
    context_name=$(KUBECONFIG="$config_file" kubectl config get-contexts --no-headers -o name 2>/dev/null | head -1)
    if [ -n "$context_name" ]; then
        contexts_to_replace+=("$context_name")
        echo "Will add/update context: $context_name (from $(basename "$config_file"))"
    fi
done

# Start with existing config if it exists
if [ -f "${USER_HOME}/.kube/config" ]; then
    cp "${USER_HOME}/.kube/config" "${STAGING_DIR}/config.bak.yaml"
    
    # Count existing contexts for reporting
    existing_context_count=$(KUBECONFIG="${STAGING_DIR}/config.bak.yaml" kubectl config get-contexts --no-headers 2>/dev/null | wc -l)
    echo "Starting with existing kubeconfig ($existing_context_count contexts)"
    
    # Remove ONLY the contexts that will be replaced (preserve all others)
    for context_name in "${contexts_to_replace[@]}"; do
        # Check if this context exists in the current config
        if KUBECONFIG="${STAGING_DIR}/config.bak.yaml" kubectl config get-contexts "$context_name" --no-headers >/dev/null 2>&1; then
            # Get cluster and user associated with this context before deleting
            KUBECONFIG="${STAGING_DIR}/config.bak.yaml" kubectl config get-contexts "$context_name" --no-headers 2>/dev/null | \
            while read -r current name cluster authinfo namespace; do
                if [ -n "$cluster" ]; then
                    echo "  Removing old cluster: $cluster"
                    KUBECONFIG="${STAGING_DIR}/config.bak.yaml" kubectl config unset "clusters.$cluster" >/dev/null 2>&1 || true
                fi
                if [ -n "$authinfo" ]; then
                    echo "  Removing old user: $authinfo"
                    KUBECONFIG="${STAGING_DIR}/config.bak.yaml" kubectl config unset "users.$authinfo" >/dev/null 2>&1 || true
                fi
            done
            
            echo "  Removing old context: $context_name"
            KUBECONFIG="${STAGING_DIR}/config.bak.yaml" kubectl config unset "contexts.$context_name" >/dev/null 2>&1 || true
        else
            echo "  Context $context_name is new (not in existing config)"
        fi
    done
    
    config_files=("${STAGING_DIR}/config.bak.yaml")
else
    echo "No existing kubeconfig found, creating fresh config"
    config_files=()
fi

# Add new configs
config_files+=("${new_configs[@]}")

# Merge all configs
OLD_IFS="$IFS"
IFS=':'
export KUBECONFIG="${config_files[*]}"
IFS="$OLD_IFS"

# Generate merged config
if ! kubectl config view --flatten > "${STAGING_DIR}/merged-${USER}.yaml" 2>/dev/null; then
    echo "Error: Failed to generate merged config with kubectl"
    echo "KUBECONFIG was: $KUBECONFIG"
    exit 1
fi

# Ensure user's .kube directory exists
mkdir -p "${USER_HOME}/.kube"

# Replace user's config with new merged version
cp "${STAGING_DIR}/merged-${USER}.yaml" "${USER_HOME}/.kube/config"
chown "${USER}":"${USER}" "${USER_HOME}/.kube/config"
chmod 600 "${USER_HOME}/.kube/config"

# Count contexts for reporting
context_count=$(kubectl --kubeconfig="${USER_HOME}/.kube/config" config get-contexts --no-headers 2>/dev/null | wc -l)
echo ""
echo "✓ Generated kubeconfig for ${USER} with ${context_count} total contexts (${#new_configs[@]} updated)"

# Show all contexts
echo ""
echo "Available contexts:"
kubectl --kubeconfig="${USER_HOME}/.kube/config" config get-contexts 2>/dev/null || echo "Failed to list contexts"

# Cleanup temp file
rm -f "${STAGING_DIR}/config.bak.yaml"