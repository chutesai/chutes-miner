#!/bin/sh
# Failed chute cleanup: delete Failed pods and orphaned chute Services (no matching Job).
# Runs in-cluster with service account; expects KUBECONFIG or in-cluster config.
set -e

NAMESPACE="${NAMESPACE:-chutes}"

kubectl delete pods --field-selector=status.phase=Failed -l chutes/chute=true -n "$NAMESPACE" 2>/dev/null || true

for svc in $(kubectl get svc -n "$NAMESPACE" -l chutes/chute=true -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
  [ -z "$svc" ] && continue
  deployment_id=$(kubectl get svc "$svc" -n "$NAMESPACE" -o jsonpath='{.spec.selector.chutes/deployment-id}' 2>/dev/null)
  if [ -n "$deployment_id" ] && ! kubectl get job -n "$NAMESPACE" -l chutes/deployment-id="${deployment_id}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null | grep -q .; then
    kubectl delete svc "$svc" -n "$NAMESPACE"
  fi
done
