from enum import Enum


class KubernetesResourceType(Enum):
    """Enum for Kubernetes resource types"""

    # Core resources
    POD = "Pod"
    SERVICE = "Service"
    CONFIG_MAP = "ConfigMap"
    SECRET = "Secret"
    PERSISTENT_VOLUME_CLAIM = "PersistentVolumeClaim"
    NAMESPACE = "Namespace"

    # Apps resources
    DEPLOYMENT = "Deployment"
    REPLICA_SET = "ReplicaSet"
    STATEFUL_SET = "StatefulSet"
    DAEMON_SET = "DaemonSet"

    # Batch resources
    JOB = "Job"
    CRON_JOB = "CronJob"

    # Networking resources
    INGRESS = "Ingress"
    NETWORK_POLICY = "NetworkPolicy"

    # RBAC resources
    ROLE = "Role"
    ROLE_BINDING = "RoleBinding"
    SERVICE_ACCOUNT = "ServiceAccount"

    # Cluster-scoped resources
    NODE = "Node"
    PERSISTENT_VOLUME = "PersistentVolume"
    CLUSTER_ROLE = "ClusterRole"
    CLUSTER_ROLE_BINDING = "ClusterRoleBinding"
    STORAGE_CLASS = "StorageClass"

    @classmethod
    def from_string(cls, resource_type_str: str) -> "KubernetesResourceType":
        """Convert string to enum, with fallback for unknown types"""
        for resource_type in cls:
            if resource_type.value == resource_type_str:
                return resource_type
        raise ValueError(f"Unknown resource type: {resource_type_str}")

    @property
    def is_namespaced(self) -> bool:
        """Check if this resource type is namespaced"""
        cluster_scoped = {
            self.NODE,
            self.PERSISTENT_VOLUME,
            self.CLUSTER_ROLE,
            self.CLUSTER_ROLE_BINDING,
            self.STORAGE_CLASS,
        }
        return self not in cluster_scoped
