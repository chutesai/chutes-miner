"""
Shared server registration logic.

Consumed by:
- Miner API (add-node flow) - uses track_server for local DB state
- Registration API (standalone) - uses check_validator_inventory for remote state

Verification and bootstrap flow are shared. Server tracking (DB, monitoring)
lives in api.server.util (miner API only).
"""

from chutes_miner.common.bootstrap import bootstrap_server, verify_server
from chutes_miner.common.inventory import (
    InventoryCheckResult,
    check_validator_inventory,
)
from chutes_miner.common.verification import (
    GravalVerificationStrategy,
    TEEVerificationStrategy,
    VerificationStrategy,
)

__all__ = [
    "bootstrap_server",
    "verify_server",
    "check_validator_inventory",
    "InventoryCheckResult",
    "GravalVerificationStrategy",
    "TEEVerificationStrategy",
    "VerificationStrategy",
]
