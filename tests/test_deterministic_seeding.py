import re

from convobench.scenarios.relay import NoisyRelay


def _extract_priority_json_block(text: str) -> str:
    # We don't parse JSON; we just compare stable substrings deterministically.
    m = re.search(r"PRIORITY TASK:\n(\{.*?\})\n", text, re.DOTALL)
    return m.group(1) if m else ""


def test_noisy_relay_seed_produces_same_noise():
    s1 = NoisyRelay(chain_length=3, noise_level="medium")
    s1.set_seed(123)
    i1 = s1.create_instance()

    s2 = NoisyRelay(chain_length=3, noise_level="medium")
    s2.set_seed(123)
    i2 = s2.create_instance()

    assert i1.initial_message.content == i2.initial_message.content


def test_noisy_relay_different_seed_changes_noise():
    s1 = NoisyRelay(chain_length=3, noise_level="medium")
    s1.set_seed(123)
    i1 = s1.create_instance()

    s2 = NoisyRelay(chain_length=3, noise_level="medium")
    s2.set_seed(456)
    i2 = s2.create_instance()

    assert i1.initial_message.content != i2.initial_message.content
