# Node bootstrapping

> **This playbook provisions the control-plane node only.** GPU **worker** nodes are now
> TEE (Intel TDX confidential VM) servers provisioned by the separate
> [`sek8s`](https://github.com/chutesai/sek8s/tree/main/host-tools) repo — they are not bootstrapped
> here. See the [main README step 8](../../README.md#8-deploy-and-add-your-tee-worker-nodes) for the
> worker workflow. The one worker-related task this playbook performs is federating TEE workers into
> monitoring (see [below](#federate-tee-confidential-vm-servers-into-monitoring)).

To ensure the highest probability of success, you should provision your control-plane server with `Ubuntu 22.04`.

## 📋 Table of Contents

- [Node Bootstrapping](#node-bootstrapping)
  - [Networking Note Before Starting](#networking-note-before-starting)
    - [External IP](#external_ip)
- [1. Install Ansible](#1-install-ansible)
  - [Mac](#mac)
  - [Ubuntu/Ubuntu (WSL)/Aptitude Based Systems](#ubuntuubuntu-wslaptitude-based-systems)
  - [CentOS/RHEL/Fedora](#centosrhelfedora)
- [2. Install Ansible Collections](#2-install-ansible-collections)
- [Optional: Performance Tweaks for Ansible](#optional-performance-tweaks-for-ansible)
- [3. Update Configuration](#3-update-configuration)
- [4. Bootstrap the Nodes](#4-bootstrap-the-nodes)
  - [Bootstrap](#bootstrap)
- [To Add a New Node, After the Fact](#to-add-a-new-node-after-the-fact)
- [Update Charts](#update-charts)
- [Restart K8s Resources](#restart-k8s-resources)

### Networking note before starting!!!

#### external_ip

> This networking requirement applies to your GPU **worker** (TEE) hosts, which are provisioned by
> [sek8s](https://github.com/chutesai/sek8s/tree/main/host-tools) — it is documented here for
> reference. Configure it on the worker hosts per the sek8s docs, not via this playbook.

Every GPU worker functions as a standalone cluster. The chutes API/validator sends traffic directly to each GPU worker, and does not route through the main CPU node at all. For the system to work, this means each GPU worker host must have a publicly routeable IP address that is not behind a shared IP (since it uses kubernetes nodePort services). This IP is the public IPv4, and must not be something in the private IP range like 192.168.0.0/16, 10.0.0.0/8, etc.

This public IP *must* be dedicated, and be the same for both egress and ingress. This means, for a node to pass validation, when the validator connects to it, the IP address you advertise as a miner must match the IP address the validator sees when your node fetches a remote token, i.e. you can't use a shared IP with NAT/port-mapping if the underlying nodes route back out to the internet with some other IPs.

## 1. Install ansible (on your local system, not the miner node(s))

### Mac

If you haven't yet, setup homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install ansible:
```bash
brew install ansible
```

### Ubuntu/Ubuntu (WSL)/aptitude based systems

```bash
sudo apt -y update && sudo apt -y install ansible python3-pip
```

### CentOS/RHEL/Fedora

Install epel repo if you haven't (and it's not fedora)
```bash
sudo dnf install epel-release -y
```

Install ansible:
```bash
sudo dnf install ansible -y
```

## 2. Install ansible collections

```bash
ansible-galaxy collection install community.general
ansible-galaxy collection install kubernetes.core
ansible-galaxy collection install ansible.posix
```

## OPTIONAL: Performance Tweaks for Ansible 

```bash
wget https://files.pythonhosted.org/packages/source/m/mitogen/mitogen-0.3.22.tar.gz
tar -xzf mitogen-0.3.22.tar.gz
```

Then in your ansible.cfg

```
[defaults]
strategy_plugins = /path/to/mitogen-0.3.22/ansible_mitogen/plugins/strategy
strategy = mitogen_linear
... leave the rest, and add this block below
[ssh_connection]
ssh_args = -o ControlMaster=auto -o ControlPersist=2m
```

## 3. Update Configuration

If you haven't already gone through the local configuratoin setup, go setup your local inventory and values according to the [pre-requisites](../../README.md#2-configure-prerequisites).

## 4. Bootstrap the nodes

### Bootstrap

Ansible handles the full setup providing you have configured the variables correctly.  It will configure the host, create the necessary k8s secrets for authentication and deploy the charts.

Execute the playbook from the `ansible/k3s` directory.

```bash
ansible-playbook -i ~/chutes/inventory.yml playbooks/site.yml
```

## To add a new node, after the fact

> **Note:** GPU **worker** nodes are now TEE (Intel TDX confidential VM) servers provisioned by the
> separate [`sek8s`](https://github.com/chutesai/sek8s/tree/main/host-tools) repo — they are **not**
> added with this playbook. Use the steps below only for additional control-plane / legacy non-TEE
> nodes. To onboard a TEE worker, deploy it via sek8s, federate it into monitoring (see
> [below](#federate-tee-confidential-vm-servers-into-monitoring)), then register it with
> `chutes-miner add-node`.

First, update your inventory.yml with the new host configuration.

Then, use the `site.yml` playbook to add the new node:
```bash
ansible-playbook -i ~/chutes/inventory.yml playbooks/site.yml --tags add-nodes
```

## Update charts

If you need to update charts for any reason, you can just use the `deploy-charts` playbook

To update all charts
```bash
ansible-playbook -i ~/chutes/inventory.yml playbooks/deploy-charts.yml
```

To update specific charts
```bash
ansible-playbook -i ~/chutes/inventory.yml playbooks/deploy-charts.yml --tags miner-charts
ansible-playbook -i ~/chutes/inventory.yml playbooks/deploy-charts.yml --tags monitoring-charts
```

> The `miner-gpu-charts` tag targets the GPU worker cluster and is **legacy** — TEE workers get
> their in-VM charts from [sek8s](https://github.com/chutesai/sek8s/tree/main/host-tools), so it is
> not used when provisioning the control plane here.

## Federate TEE (confidential VM) servers into monitoring

TEE servers run k3s inside a confidential VM and are provisioned by a separate
repo (`sek8s`), so they are **not** configured by this playbook. They only need
to be federated into the control-plane Prometheus so their metrics show up in
Grafana.

Inside the VM, Prometheus is already exposed on NodePort `30090`, and the host
bridge forwards the NodePort range (`30000-32767`) to the VM — so the TEE *host*
public IP at port `30090` reaches the VM's Prometheus with no extra setup.

Add the TEE hosts to a `tee_workers` group. Because you typically keep these in
a separate inventory (managed by the other repo), the cleanest approach is a
small dedicated inventory file:

```yaml
# tee-inventory.yml
all:
  children:
    tee_workers:
      hosts:
        chutes-miner-tee-0:
          ansible_host: <tee-host-public-ip>   # host that bridges to the VM
          # federation_port: 30090             # override only if non-default
```

Then merge it in for the monitoring deploy only (the provisioning plays in
`site.yml` exclude `tee_workers`, so TEE hosts are never touched here):

```bash
ansible-playbook -i ~/chutes/inventory.yml -i ~/chutes/tee-inventory.yml \
  playbooks/deploy-charts.yml --tags monitoring-charts
```

## Restart K8s Resources
To restart deployments and daemonsets across all clusters:

```
ansible-playbook -i ~/chutes/inventory.yml playbooks/restart-k8s.yml
```