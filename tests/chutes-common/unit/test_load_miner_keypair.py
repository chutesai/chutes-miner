import pytest
from substrateinterface import Keypair

from chutes_common.settings import load_miner_keypair

SEED = "0xe031170f32b4cda05df2f3cf6bc8d7687b683bbce23d9fa960c0b3fc21641b8a"
EXPECTED = Keypair.create_from_seed(SEED)


@pytest.fixture(autouse=True)
def miner_ss58(monkeypatch):
    monkeypatch.setenv("MINER_SS58", EXPECTED.ss58_address)


def test_load_miner_keypair_from_seed(monkeypatch):
    monkeypatch.delenv("MINER_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("MINER_SEED", SEED)
    assert load_miner_keypair().ss58_address == EXPECTED.ss58_address


def test_load_miner_keypair_from_private_key(monkeypatch):
    monkeypatch.delenv("MINER_SEED", raising=False)
    monkeypatch.setenv("MINER_PRIVATE_KEY", EXPECTED.private_key.hex())
    keypair = load_miner_keypair()
    assert keypair.ss58_address == EXPECTED.ss58_address
    assert EXPECTED.verify(b"payload", keypair.sign(b"payload"))


def test_load_miner_keypair_prefers_private_key(monkeypatch):
    other = Keypair.create_from_seed("0x" + "11" * 32)
    monkeypatch.setenv("MINER_SS58", other.ss58_address)
    monkeypatch.setenv("MINER_SEED", SEED)
    monkeypatch.setenv("MINER_PRIVATE_KEY", other.private_key.hex())
    assert load_miner_keypair().ss58_address == other.ss58_address


def test_load_miner_keypair_rejects_mismatched_ss58(monkeypatch):
    other = Keypair.create_from_seed("0x" + "11" * 32)
    monkeypatch.delenv("MINER_SEED", raising=False)
    monkeypatch.setenv("MINER_PRIVATE_KEY", other.private_key.hex())
    with pytest.raises(ValueError, match="does not match MINER_SS58"):
        load_miner_keypair()
