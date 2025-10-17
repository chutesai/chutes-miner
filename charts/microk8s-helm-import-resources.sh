#!/bin/bash

# Configuration - Hardcoded for limited context
NAMESPACE="chutes"
CHART_PATH1="./chutes-miner"
RELEASE_NAME1="chutes"
CHART_PATH2="./chutes-miner-gpu"
RELEASE_NAME2="chutes-gpu"

# Values file as argument
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/values.yaml"
    exit 1
fi
VALUES_FILE="$1"

# Function to process a single chart
process_chart() {
    local CHART_PATH="$1"
    local RELEASE_NAME="$2"

    # Generate templated manifests
    echo "Generating templated manifests for $CHART_PATH..."
    TEMP_DIR=$(mktemp -d)
    microk8s helm template "$RELEASE_NAME" "$CHART_PATH" -f "$VALUES_FILE" -n "$NAMESPACE" --output-dir "$TEMP_DIR"

    # Extract unique kind/name pairs (improved parsing for YAML structure)
    RESOURCES=$(find "$TEMP_DIR" -name "*.yaml" -exec cat {} + | awk '
    BEGIN { kind=""; name=""; }
    /^kind:/ { kind=$2; }
    /^  name:/ { name=$2; if (kind != "") { print kind " " name; kind=""; name=""; } }
    ' | sort -u)

    # Patch each resource
    echo "Patching resources for $RELEASE_NAME with Helm ownership metadata..."
    while read -r kind name; do
        if [ -n "$kind" ] && [ -n "$name" ]; then
            if [ $kind = "PriorityClass" ]; then
                continue
            fi
            
            echo "Patching $kind/$name..."
            
            # Add annotations
            microk8s kubectl annotate "$kind" "$name" \
                "meta.helm.sh/release-name=$RELEASE_NAME" \
                "meta.helm.sh/release-namespace=$NAMESPACE" \
                --namespace="$NAMESPACE" \
                --overwrite
            
            # Add label
            microk8s kubectl label "$kind" "$name" \
                "app.kubernetes.io/managed-by=Helm" \
                --namespace="$NAMESPACE" \
                --overwrite

            echo "  ✓ Done"
        fi
    done <<< "$RESOURCES"

    # Cleanup
    rm -rf "$TEMP_DIR"
}

# Process both charts
process_chart "$CHART_PATH1" "$RELEASE_NAME1"
process_chart "$CHART_PATH2" "$RELEASE_NAME2"

echo "Patching complete! Now run your Helm upgrades:"
echo "microk8s helm upgrade --install $RELEASE_NAME1 $CHART_PATH1 -f $VALUES_FILE --namespace $NAMESPACE"
echo "microk8s helm upgrade --install $RELEASE_NAME2 $CHART_PATH2 -f $VALUES_FILE --namespace $NAMESPACE"