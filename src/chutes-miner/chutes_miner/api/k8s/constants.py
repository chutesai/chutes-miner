SEARCH_NODES_PATH = "/apis/search.karmada.io/v1alpha1/search/cache/api/v1/nodes"
SEARCH_PODS_PATH = "/apis/search.karmada.io/v1alpha1/search/cache/api/v1/pods"
SEARCH_SERVICES_PATH = "/apis/search.karmada.io/v1alpha1/search/cache/api/v1/services"
SEARCH_DEPLOYMENTS_PATH = "/apis/search.karmada.io/v1alpha1/search/cache/apis/apps/v1/deployments"

CHUTE_CODE_CM_PREFIX = "chute-code"
CHUTE_SVC_PREFIX = "chute-service"
CHUTE_DEPLOY_PREFIX = "chute"
CHUTE_PP_PREFIX = "chute-pp"
GRAVAL_PP_PREFIX = "graval-pp"
GRAVAL_JOB_PREFIX = "graval"
GRAVAL_SVC_PREFIX = "graval-svc"

# Port discovery: well-known service names for NodePort lookup
AGENT_SERVICE_NAME = "agent"
ATTESTATION_SERVICE_NAME = "attestation-service-external"
ATTESTATION_NAMESPACE = "attestation-system"
ATTESTATION_PORT_NAME = "https-external"  # ServicePort.name for attestation HTTPS (8443)
DEFAULT_ATTESTATION_PORT = 30443
DEFAULT_AGENT_PORT = 32000
