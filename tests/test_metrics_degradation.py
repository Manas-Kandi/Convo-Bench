import json

from convobench.metrics.degradation import (
    constraint_metrics,
    entity_retention,
    json_structure_preservation,
)


def test_json_structure_preservation_perfect_match():
    expected = {"a": 1, "b": {"c": "x"}}
    observed = """```json
{"a": 1, "b": {"c": "x"}}
```"""
    r = json_structure_preservation(expected, observed)
    assert r is not None
    assert r.key_f1 == 1.0
    assert r.value_exact_match_rate == 1.0


def test_json_structure_preservation_partial():
    expected = {"a": 1, "b": {"c": "x"}}
    observed = """```json
{"a": 2}
```"""
    r = json_structure_preservation(expected, observed)
    assert r is not None
    assert r.key_f1 < 1.0
    assert r.value_exact_match_rate < 1.0


def test_entity_retention_amount_date_name():
    expected = "Alice met Bob on March 15, 2025 and spent $1,200.00"
    observed = "Bob met Alice on March 15, 2025 and spent $1,200.00"
    ent = entity_retention(expected, observed)
    assert ent["names_retention"] == 1.0
    assert ent["dates_retention"] == 1.0
    assert ent["amounts_retention"] == 1.0


def test_constraint_metrics_mentions_and_violation():
    constraints = ["Do not share customer names externally", "Must not exceed 10 pages"]
    observed = "You can share customer names externally. Must not exceed 10 pages."
    c = constraint_metrics(constraints, observed)
    assert c.constraints_total == 2
    assert c.constraints_mentioned >= 1
    assert c.constraints_violated >= 1
