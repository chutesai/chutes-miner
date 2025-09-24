# Chart Migration

## Overview

If you are not migrating to k3s yet, you should at least migrate your charts to the latest versions in order to ensure you can apply any updates that are provided.

## Migration Process

### Set Values for single cluster

```yaml
# ~/chutes/values.yaml
multiCluster: false
```

### Update K8s resources to be managed by helm

If you previously deployed using static manifests, you need to import the resources so helm can managed them.  You can use the helper script `charts/helm-import-resources.sh` to import them.

Import resources:
```bash
cd charts
./helm-import-resources.sh
cd ..
```

Update gepetto:
```bash
kubectl create configmap gepetto-code --from-file=$HOME/chutes/gepetto.py -o yaml --dry-run=client | kubectl apply -n chutes -f -
```

Deploy `chutes-miner` charts for control node components
```bash
helm upgrade --insall chutes charts/chutes-miner -f values ~/chutes/values.yaml
```

Deploy `chutes-miner-gpu` charts for control node components
```bash
helm upgrade --insall chutes-gpu charts/chutes-miner-gpu -f values ~/chutes/values.yaml
```