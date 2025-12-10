import sys
import types
from collections import defaultdict

import pandas as pd
from filelock import FileLock

# Provide a lightweight tensorflow stub to satisfy imports without heavyweight dependency
tf_stub = types.ModuleType("tensorflow")
tf_stub.keras = types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda: None))
sys.modules.setdefault("tensorflow", tf_stub)

import scripts.export_annotations_with_edit_distances as exp


def _run_process_with_flags(tmp_path, include_edit, include_seqs, fake_tf):
    # Minimal annotated read payload
    annotated_reads = [{
        'read_length': 4,
        'read': 'ACTG',
        'architecture': 'valid',
        'reason': 'ok',
        'orientation': '+',
        'CBC': {'Starts': [0], 'Ends': [3], 'Sequences': ['ACT']},
        'cDNA': {'Starts': [0], 'Ends': [4]},
        'UMI': {'Starts': [0], 'Ends': [2]},
    }]

    def fake_extract(*_args, **_kwargs):
        return annotated_reads

    def fake_whitelists(*_args, **_kwargs):
        return ({'CBC': ['ACT']}, {}) if include_edit else ({}, {})

    def fake_bc(valid_reads_df, _strand, barcode_columns, *_rest):
        corrected = pd.DataFrame({
            'ReadName': valid_reads_df['ReadName'],
            'read_length': valid_reads_df['read_length'],
            'architecture': valid_reads_df['architecture'],
            'reason': valid_reads_df['reason'],
            'orientation': valid_reads_df['orientation'],
        })
        for barcode in barcode_columns:
            corrected[f'corrected_{barcode}_counts_with_min_dist'] = [1] * len(corrected)
            corrected[f'corrected_{barcode}_min_dist'] = [0] * len(corrected)
        return corrected, defaultdict(int), defaultdict(int)

    fake_tf.setattr(exp, "extract_annotated_full_length_seqs", fake_extract)
    fake_tf.setattr(exp, "get_or_load_whitelists_and_sequences", fake_whitelists)
    fake_tf.setattr(exp, "bc_n_demultiplex", fake_bc)

    config = exp.AnnotateReadsConfig(
        output_fmt="fasta",
        model_type="REG",
        pass_num=1,
        model_path_w_CRF="",
        threshold=0,
        n_jobs=1,
        include_edit_distances=include_edit,
        include_sequences_in_valid_output=include_seqs,
    )

    output_dir = tmp_path
    valid_output_file = output_dir / "valid.tsv"
    invalid_output_file = output_dir / "invalid.tsv"

    exp.process_full_length_reads_in_chunks_and_save(
        config=config,
        reads=["ACTG"],
        original_read_names=["read1"],
        strand="fwd",
        base_qualities=["!!!!"],
        predictions=None,
        bin_name="bin1",
        chunk_idx=1,
        label_binarizer=None,
        cumulative_barcodes_stats={'CBC': {'count_data': {}, 'min_dist_data': {}}},
        actual_lengths=[4],
        seq_order=[],
        add_header=True,
        output_dir=str(output_dir),
        invalid_output_file=str(invalid_output_file),
        invalid_file_lock=FileLock(str(invalid_output_file) + ".lock"),
        valid_output_file=str(valid_output_file),
        valid_file_lock=FileLock(str(valid_output_file) + ".lock"),
        barcodes=["CBC"],
        whitelist_df=pd.DataFrame(),
        whitelist_dict={},
        demuxed_fasta=str(output_dir / "demux.fa"),
        demuxed_fasta_lock=FileLock(str(output_dir / "demux.fa") + ".lock"),
        ambiguous_fasta=str(output_dir / "amb.fa"),
        ambiguous_fasta_lock=FileLock(str(output_dir / "amb.fa") + ".lock"),
    )

    return pd.read_csv(valid_output_file, sep='\t').columns.tolist()


def test_process_flags_exclude_optional_columns(tmp_path, fake_tf):
    cols = _run_process_with_flags(tmp_path, include_edit=False, include_seqs=False, fake_tf=fake_tf)
    assert 'read' not in cols
    assert not any(col.endswith('_edit_distance') for col in cols)


def test_process_flags_include_optional_columns(tmp_path, fake_tf):
    cols = _run_process_with_flags(tmp_path, include_edit=True, include_seqs=True, fake_tf=fake_tf)
    assert 'read' in cols
    assert 'CBC_edit_distance' in cols
    assert 'CBC_match_orientation' in cols
