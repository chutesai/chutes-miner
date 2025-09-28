#!/bin/bash
set -eo pipefail

# Script to merge kubeconfig files, preserving existing contexts
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

# Start with existing config if available, otherwise empty
config_files=()
if [ -f "${STAGING_DIR}/existing_config_backup.yaml" ]; then
    config_files+=("${STAGING_DIR}/existing_config_backup.yaml")
    echo "Starting with existing kubeconfig (preserving non-targeted host contexts)"
fi

# Add new/updated configs from targeted hosts
new_configs_count=0
shopt -s nullglob  # Make glob return empty array if no matches
for config_file in "${STAGING_DIR}"/*.yaml; do
    if [ -f "$config_file" ] && [[ "$config_file" != *"existing_config_backup.yaml"* ]] && [[ "$config_file" != *"merged-"* ]]; then
        config_files+=("$config_file")
        new_configs_count=$((new_configs_count + 1))
        echo "Adding config: $(basename "$config_file" .yaml)"
    fi
done
shopt -u nullglob

if [ ${#config_files[@]} -gt 0 ]; then
    echo "Merging ${new_configs_count} new/updated configs with existing contexts"
    
    # Join array elements with colons for KUBECONFIG
    OLD_IFS="$IFS"
    IFS=':'
    export KUBECONFIG="${config_files[*]}"
    IFS="$OLD_IFS"
    
    # Test kubectl availability
    if ! command -v kubectl >/dev/null 2>&1; then
        echo "Error: kubectl not found in PATH"
        exit 1
    fi
    
    # Generate fresh merged config
    if ! kubectl config view --flatten > "${STAGING_DIR}/merged-${USER}.yaml" 2>/dev/null; then
        echo "Error: Failed to generate merged config with kubectl"
        echo "KUBECONFIG was: $KUBECONFIG"
        echo "Available contexts:"
        kubectl config get-contexts 2>/dev/null || echo "Failed to get contexts"
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
    echo "Generated kubeconfig for ${USER} with ${context_count} total contexts (${new_configs_count} updated)"
    
    # Show the contexts that were created
    echo "Available contexts:"
    kubectl --kubeconfig="${USER_HOME}/.kube/config" config get-contexts --no-headers 2>/dev/null || echo "Failed to list contexts"
else
    echo "No kubeconfig files found to merge for ${USER}"
    exit 1
fi