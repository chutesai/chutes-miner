# Node bootstrapping

To ensure the highest probability of success, you should provision your servers with `Ubuntu 22.04`, preferably with NO nvidia driver installations if possible.

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

Every GPU node functions as a standalone cluster. The chutes API/validator sends traffic directly to each GPU node, and does not route through the main CPU node at all. For the system to work, this means each GPU node must have a publicly routeable IP address on each GPU node that is not behind a shared IP (since it uses kubernetes nodePort services). This IP is the public IPv4, and must not be something in the private IP range like 192.168.0.0/16, 10.0.0.0/8, etc.

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
ansible-playbook -i ~/chutes/inventory.yml playbooks/deploy-charts.yml --tags miner-gpu-charts
ansible-playbook -i ~/chutes/inventory.yml playbooks/deploy-charts.yml --tags monitoring-charts
```

## Restart K8s Resources
To restart deployments and daemonsets across all clusters:

```
ansible-playbook -i ~/chutes/inventory.yml playbooks/restart-k8s.yml
```