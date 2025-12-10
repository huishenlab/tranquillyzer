from pathlib import Path

import pytest

from main import parse_whitelist_arg
import scripts.export_annotations_with_edit_distances as exp


def reset_whitelist_cache():
    exp._CACHED_WHITELISTS = None
    exp._CACHED_KNOWN_SEQUENCES = None
    exp._CURRENT_CACHED_SAMPLE_ID = None
    exp._CACHED_WHITELIST_PATHS = None


def write_seq_orders(tmp_path: Path, model: str, segments: str, sequences: str = "AAA,BBB"):
    seq_orders = tmp_path / "seq_orders.tsv"
    seq_orders.write_text(f'{model}\t"{segments}"\t"{sequences}"\n')
    return seq_orders


def test_parse_whitelist_arg_roundtrip(tmp_path):
    path1 = tmp_path / "a.txt"
    path2 = tmp_path / "b.txt"
    result = parse_whitelist_arg(f"seg1:{path1},seg2:{path2}")
    assert result == {"seg1": str(path1), "seg2": str(path2)}

    with pytest.raises(ValueError):
        parse_whitelist_arg("missingcolon")


def test_explicit_whitelists_respect_seq_order(tmp_path):
    # Use variable-length patterns so segments remain eligible for whitelist lookup
    seq_orders_path = write_seq_orders(tmp_path, "modelA", "seg1,seg2", sequences="NN,NN")

    # Write whitelist only for seg1; segX should be ignored because it's not in seq_orders
    seg1_file = tmp_path / "seg1.txt"
    seg1_file.write_text("id\tACT\n")
    segx_file = tmp_path / "segX.txt"
    segx_file.write_text("id\tAAA\n")

    reset_whitelist_cache()
    whitelists, known = exp.get_or_load_whitelists_and_sequences(
        seq_orders_path=str(seq_orders_path),
        model_name="modelA",
        whitelist_df=None,
        metadata_path=None,
        sample_id=None,
        whitelist_paths={"seg1": str(seg1_file), "segX": str(segx_file)},
    )

    assert whitelists.get("seg1") == ["ACT"]
    assert "segX" not in whitelists
