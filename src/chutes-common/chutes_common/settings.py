import os
import json
from typing import List
from pydantic import BaseModel, Field
from substrateinterface import Keypair, KeypairType
from pydantic_settings import BaseSettings


def load_miner_keypair() -> Keypair:
    """
    Load the miner keypair from MINER_PRIVATE_KEY (64-byte sr25519 private key, as stored
    in newer bittensor-wallet hotkeys) if set, otherwise from MINER_SEED (32-byte seed).
    """
    private_key = os.getenv("MINER_PRIVATE_KEY")
    if private_key:
        return Keypair.create_from_private_key(
            private_key, ss58_format=42, crypto_type=KeypairType.SR25519
        )
    return Keypair.create_from_seed(os.environ["MINER_SEED"])


class Validator(BaseModel):
    hotkey: str
    registry: str
    api: str
    socket: str


class MinerSettings(BaseSettings):
    _validators: List[Validator] = []

    miner_ss58: str = os.environ["MINER_SS58"]
    miner_keypair: Keypair = load_miner_keypair()
    validators_json: str = os.environ["VALIDATORS"]

    @property
    def validators(self) -> List[Validator]:
        if self._validators:
            return self._validators
        data = json.loads(self.validators_json)
        self._validators = [Validator(**item) for item in data["supported"]]
        return self._validators


miner_settings = MinerSettings()


class RedisSettings(BaseSettings):
    redis_url: str = Field(default="redis://redis:6379", description="Redis URL")


# redis_settings = RedisSettings()
