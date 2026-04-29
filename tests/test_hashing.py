from baroque.core.hashing import canonical_json, content_hash


def test_canonical_json_sorts_mapping_keys() -> None:
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_content_hash_is_stable_for_equivalent_mappings() -> None:
    left = content_hash({"b": [2, 3], "a": 1})
    right = content_hash({"a": 1, "b": [2, 3]})
    assert left == right
    assert left.startswith("sha256:")

