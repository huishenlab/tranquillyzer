import os
import re
import gc
import csv
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import tensorflow as tf
from dataclasses import dataclass
from filelock import FileLock
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
from typing import Dict, List, Optional, Tuple

from matplotlib.backends.backend_pdf import PdfPages
from scripts.correct_barcodes import bc_n_demultiplex
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class AnnotateReadsConfig:
    """Configuration for read annotation processing."""
    output_fmt: str
    model_type: str
    pass_num: int
    model_path_w_CRF: str
    threshold: int
    n_jobs: int
    seq_orders_path: Optional[str] = None
    model_name: Optional[str] = None
    whitelist_paths: Optional[Dict[str, str]] = None
    metadata_path: Optional[str] = None
    sample_id: Optional[str] = None
    include_edit_distances: bool = False
    include_sequences_in_valid_output: bool = False


# Constants
MAX_SEQ_DISPLAY_LENGTH = 50


# Edit Distance Helper Functions

def reverse_complement(seq):
    """Reverse complement a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join([complement.get(base, 'N') for base in seq[::-1]])


def load_whitelist_sequences(whitelist_path):
    """Load sequences from whitelist TSV file (ID\tSequence format)."""
    sequences = []
    if os.path.exists(whitelist_path):
        with open(whitelist_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sequences.append(parts[1])
    return sequences


def calculate_min_edit_distance(detected_seq, reference_seqs, check_revcomp=True):
    """
    Calculate the minimum Levenshtein distance between a detected sequence and reference(s).

    Args:
        detected_seq (str): Observed sequence string.
        reference_seqs (str | list[str]): Reference sequence or list of reference sequences.
        check_revcomp (bool): If True, also compare reverse complement of detected_seq.

    Returns:
        tuple[int | None, str | None]: (minimum edit distance, orientation), where orientation
        is 'fwd', 'rev', or None when no references are provided.
    """
    if not detected_seq or not reference_seqs:
        return (None, None)

    # Handle single reference sequence
    if isinstance(reference_seqs, str):
        reference_seqs = [reference_seqs]

    min_dist = float('inf')
    best_orientation = None

    # Check forward orientation
    for ref_seq in reference_seqs:
        dist = Levenshtein.distance(detected_seq, ref_seq)
        if dist < min_dist:
            min_dist = dist
            best_orientation = 'fwd'

    # Check reverse complement if requested
    if check_revcomp:
        detected_rc = reverse_complement(detected_seq)
        for ref_seq in reference_seqs:
            dist = Levenshtein.distance(detected_rc, ref_seq)
            if dist < min_dist:
                min_dist = dist
                best_orientation = 'rev'

    return (min_dist if min_dist != float('inf') else None, best_orientation)


def load_whitelist_paths(whitelist_paths: Dict[str, str], segments: List[str]) -> Dict[str, List[str]]:
    """
    Load whitelists from explicit segment:path mappings, restricted to provided segments list.

    Args:
        whitelist_paths: Mapping of segment name -> file path.
        segments: Segment names allowed for loading (typically from seq_orders).

    Returns:
        dict[str, list[str]]: Loaded whitelist sequences keyed by segment.
    """
    whitelists = {}
    allowed = set(segments)

    for segment, path in whitelist_paths.items():
        if segment not in allowed:
            logger.info(f"Ignoring whitelist for '{segment}' (not present in model seq order)")
            continue

        if os.path.exists(path):
            whitelists[segment] = load_whitelist_sequences(path)
            if whitelists[segment]:
                logger.info(f"Loaded {len(whitelists[segment])} sequences for {segment} from {path}")
            else:
                logger.warning(f"No sequences loaded for {segment} from {path}")
        else:
            logger.warning(f"Whitelist file not found for {segment}: {path}")

    return whitelists


def parse_seq_orders_for_known_sequences(seq_orders_path: str, model_name: str) -> Dict[str, str]:
    """Parse seq_orders.tsv to get known sequences for a model."""
    known_seqs = {}

    if not os.path.exists(seq_orders_path):
        logger.warning(f"seq_orders file not found: {seq_orders_path}")
        return known_seqs

    with open(seq_orders_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3 and parts[0] == model_name:
                seg_order = parts[1].strip('"').split(',')
                # Parse sequences
                seqs = parts[2].strip('"').split(',')

                for label, seq in zip(seg_order, seqs):
                    # Skip variable length segments (patterns with N)
                    # But include them for reference
                    if seq not in ['NN', 'A', 'T']:
                        known_seqs[label] = seq
                        # Log fixed sequences only (no N's)
                        if 'N' not in seq:
                            logger.info(f"Loaded known sequence for {label}: {seq[:20]}{'...' if len(seq) > 20 else ''}")

                break

    return known_seqs


def load_sequences_from_metadata(metadata_path: Optional[str],
                                 sample_id: str,
                                 model_segments: List[str]) -> Dict[str, List[str]]:
    """Parse metadata TSV (Sample_id\tCBC\ti7...) for sample-specific barcode sequences."""
    if not metadata_path or not os.path.exists(metadata_path):
        if metadata_path:
            logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    try:
        df = pd.read_csv(metadata_path, sep='\t', dtype=str, comment='#')
        df.columns = df.columns.str.strip()

        sample_col = next((col for col in df.columns if col.lower() == 'sample_id'), None)
        if not sample_col:
            logger.error("Metadata file missing 'Sample_id' column")
            return {}

        df[sample_col] = df[sample_col].str.strip()
        sample_row = df[df[sample_col] == str(sample_id).strip()]

        if sample_row.empty:
            available_samples = df[sample_col].tolist()
            logger.warning(
                f"Sample ID '{sample_id}' not found in metadata. "
                f"Available samples: {', '.join(available_samples[:10])}"
                f"{'...' if len(available_samples) > 10 else ''}"
            )
            return {}

        sample_sequences = {}
        row_dict = sample_row.iloc[0].to_dict()

        for segment in model_segments:
            segment_col = next(
                (col for col in row_dict.keys() if col.lower() == segment.lower()),
                None
            )

            if segment_col and pd.notna(row_dict[segment_col]):
                seq_val = str(row_dict[segment_col]).strip()
                if not seq_val:
                    continue

                if ',' in seq_val:
                    sequences = [s.strip() for s in seq_val.split(',') if s.strip()]
                    sample_sequences[segment] = sequences
                else:
                    sample_sequences[segment] = [seq_val]

                logger.info(
                    f"Loaded {len(sample_sequences[segment])} sequence(s) for {segment} "
                    f"from metadata (Sample: {sample_id})"
                )

        return sample_sequences

    except Exception as e:
        logger.error(f"Error parsing metadata file: {e}", exc_info=True)
        return {}


# Cache for loaded whitelists and sequences
_CACHED_WHITELISTS = None
_CACHED_KNOWN_SEQUENCES = None
_CURRENT_CACHED_SAMPLE_ID = None
_CACHED_WHITELIST_PATHS = None


def get_or_load_whitelists_and_sequences(
    seq_orders_path: Optional[str] = None,
    model_name: Optional[str] = None,
    whitelist_df: Optional[pd.DataFrame] = None,
    metadata_path: Optional[str] = None,
    sample_id: Optional[str] = None,
    whitelist_paths: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Load whitelists and known sequences (cached).

    Priority: 1) Metadata (sample-specific), 2) explicit whitelist_paths,
    3) whitelist DataFrame.

    Args:
        seq_orders_path: Path to seq_orders.tsv for model segments.
        model_name: Model key used to select seq order row.
        whitelist_df: Optional DataFrame containing whitelist columns.
        metadata_path: Optional metadata TSV for sample-specific barcodes.
        sample_id: Sample identifier to lookup in metadata.
        whitelist_paths: Explicit mapping of segment->whitelist file.

    Returns:
        tuple(dict[str, list[str]], dict[str, str]): (whitelists by segment, fixed known sequences).
    """
    global _CACHED_WHITELISTS, _CACHED_KNOWN_SEQUENCES, _CURRENT_CACHED_SAMPLE_ID, _CACHED_WHITELIST_PATHS

    if sample_id != _CURRENT_CACHED_SAMPLE_ID:
        if _CURRENT_CACHED_SAMPLE_ID is not None:
            logger.info(f"Switching to Sample ID: {sample_id}. Clearing cache.")
        _CACHED_WHITELISTS = None
        _CURRENT_CACHED_SAMPLE_ID = sample_id

    if whitelist_paths is None and _CACHED_WHITELIST_PATHS is not None:
        _CACHED_WHITELISTS = None
        _CACHED_WHITELIST_PATHS = None
    elif whitelist_paths is not None and whitelist_paths != _CACHED_WHITELIST_PATHS:
        _CACHED_WHITELISTS = None
        _CACHED_WHITELIST_PATHS = dict(whitelist_paths)

    model_segments = []
    if seq_orders_path and model_name:
        if _CACHED_KNOWN_SEQUENCES is None:
            _CACHED_KNOWN_SEQUENCES = parse_seq_orders_for_known_sequences(seq_orders_path, model_name)

        with open(seq_orders_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0] == model_name:
                    model_segments = parts[1].strip('"').split(',')
                    break

    if _CACHED_WHITELISTS is None:
        _CACHED_WHITELISTS = {}

        # Priority 1: Metadata (sample-specific)
        if metadata_path and sample_id:
            metadata_seqs = load_sequences_from_metadata(metadata_path, sample_id, model_segments)
            _CACHED_WHITELISTS.update(metadata_seqs)

        remaining_segments = [s for s in model_segments if s not in _CACHED_WHITELISTS]

        # Skip fixed sequences (adapters, cDNA, etc.)
        known_seqs = _CACHED_KNOWN_SEQUENCES or {}
        skip_segments = set(known_seqs.keys()).union(['cDNA', 'NN'])
        target_segments = [s for s in remaining_segments if s not in skip_segments]

        # Priority 2: Explicit whitelist paths (segment:path)
        explicit_whitelist_paths = whitelist_paths or {}
        if explicit_whitelist_paths:
            loaded_from_paths = load_whitelist_paths(explicit_whitelist_paths, target_segments)
            _CACHED_WHITELISTS.update(loaded_from_paths)
            remaining_segments = [s for s in remaining_segments if s not in _CACHED_WHITELISTS]

        target_segments = [s for s in remaining_segments if s not in skip_segments]

        # Priority 3: DataFrame
        if whitelist_df is not None:
            for segment in target_segments:
                if segment in whitelist_df.columns:
                    sequences = whitelist_df[segment].dropna().unique().tolist()
                    sequences = [s for s in sequences if s != segment and str(s).strip() != segment]
                    _CACHED_WHITELISTS[segment] = sequences
                    logger.info(f"Loaded {len(_CACHED_WHITELISTS[segment])} sequences for {segment} from whitelist DataFrame")

    return _CACHED_WHITELISTS, _CACHED_KNOWN_SEQUENCES or {}

# Checkpointing Functions

def save_checkpoint(checkpoint_file, bin_name, chunk):
    """Write checkpoint progress to disk (bin name and chunk index)."""
    with open(checkpoint_file, "w") as f:
        f.write(f"{bin_name},{chunk}")


def load_checkpoint(checkpoint_file, start_bin):
    """Load checkpoint progress; return (bin_name, chunk_idx) or defaults if missing."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            bin_name, chunk = f.readline().strip().split(",")
        return bin_name, int(chunk)
    return start_bin, 1


# Helper Functions for Edit Distance Calculation

def _calc_segment_edit_dist(annotated_read: dict, label: str, ref_seqs) -> dict:
    """
    Calculate edit distance for a single segment.

    Args:
        annotated_read: Dict containing segment annotations incl. 'Sequences'.
        label: Segment label to evaluate.
        ref_seqs: Reference sequence(s) to compare against.

    Returns:
        dict: Keys {<label>_edit_distance, <label>_match_orientation}.
    """
    if label in annotated_read and annotated_read[label].get('Sequences'):
        detected = annotated_read[label]['Sequences'][0]
        dist, orient = calculate_min_edit_distance(detected, ref_seqs, check_revcomp=True)
        return {f'{label}_edit_distance': dist, f'{label}_match_orientation': orient}
    return {f'{label}_edit_distance': None, f'{label}_match_orientation': None}


def get_all_edit_dist_cols(annotated_read: dict, segment_whitelists: dict,
                            known_sequences: dict, barcodes: list) -> dict:
    """
    Get edit distance columns for all segments with references.

    Args:
        annotated_read: Per-read annotation dict with segment info.
        segment_whitelists: Dict of segment->list of whitelist sequences.
        known_sequences: Dict of fixed segment->sequence.
        barcodes: List of barcode segment labels (to exclude from fixed sequences).

    Returns:
        dict: Combined edit distance and orientation fields.
    """
    cols = {}

    # Process segments with whitelists
    for label, whitelist in segment_whitelists.items():
        cols.update(_calc_segment_edit_dist(annotated_read, label, whitelist))

    # Process fixed sequences (skip variable-length and barcodes)
    for label, ref_seq in known_sequences.items():
        if 'N' not in ref_seq and label not in barcodes:
            cols.update(_calc_segment_edit_dist(annotated_read, label, ref_seq))

    return cols


def build_position_cols(annotated_read: dict) -> dict:
    """
    Extract start/end position columns.

    Args:
        annotated_read: Per-read annotation dict with segment positions.

    Returns:
        dict: Keys like '<label>_Starts' and '<label>_Ends' with comma-joined indices.
    """
    starts = {
        f'{label}_Starts': ', '.join(map(str, ann['Starts']))
        for label, ann in annotated_read.items()
        if label not in {"architecture", "reason", "read", "read_length", "orientation"} and 'Starts' in ann
    }
    ends = {
        f'{label}_Ends': ', '.join(map(str, ann['Ends']))
        for label, ann in annotated_read.items()
        if label not in {"architecture", "reason", "read", "read_length", "orientation"} and 'Ends' in ann
    }
    return {**starts, **ends}


def build_sequence_cols(annotated_read: dict, barcodes: list) -> dict:
    """
    Extract barcode sequence columns.

    Args:
        annotated_read: Per-read annotation dict with sequence strings.
        barcodes: Labels treated as barcodes.

    Returns:
        dict: Keys like '<barcode>_Sequences' with comma-joined sequences.
    """
    return {
        f'{label}_Sequences': ', '.join(map(str, annotated_read[label]['Sequences']))
        for label in barcodes
        if label in annotated_read and 'Sequences' in annotated_read[label]
    }


# Main Processing Function with Edit Distances

def process_full_length_reads_in_chunks_and_save(
    config: AnnotateReadsConfig,
    reads, original_read_names,
    strand, base_qualities,
    predictions, bin_name, chunk_idx,
    label_binarizer,
    cumulative_barcodes_stats,
    actual_lengths, seq_order,
    add_header, output_dir,
    invalid_output_file,
    invalid_file_lock,
    valid_output_file,
    valid_file_lock, barcodes,
    whitelist_df, whitelist_dict,
    demuxed_fasta,
    demuxed_fasta_lock,
    ambiguous_fasta,
    ambiguous_fasta_lock
):
    """
    Process reads with edit distance calculations and write outputs.

    Args:
        config: AnnotateReadsConfig with processing toggles.
        reads (list[str]): Raw read strings for the chunk.
        original_read_names (list[str]): Read identifiers aligned with reads.
        strand (str): Library strand orientation ("fwd"/"rev").
        base_qualities (list[str] | list[None]): Base quality strings if FASTQ output.
        predictions: Model predictions for the chunk (shape matches reads).
        bin_name (str): Bin identifier for logging/checkpointing.
        chunk_idx (int): Chunk index within bin.
        label_binarizer: Label binarizer used to decode predictions.
        cumulative_barcodes_stats (dict): Running stats for barcodes.
        actual_lengths (list[int]): True read lengths per read.
        seq_order (list[str]): Model segment order.
        add_header (bool): Whether to write TSV header.
        output_dir (str): Output directory root.
        invalid_output_file (str): Path for invalid reads TSV.
        invalid_file_lock (FileLock): Lock for invalid reads output.
        valid_output_file (str): Path for valid reads TSV.
        valid_file_lock (FileLock): Lock for valid reads output.
        barcodes (list[str]): Barcode segment labels.
        whitelist_df (pd.DataFrame): Whitelist dataframe for demuxing.
        whitelist_dict (dict): Demux whitelist dictionary.
        demuxed_fasta (str): Path to demuxed FASTA/FASTQ output.
        demuxed_fasta_lock (FileLock): Lock for demuxed output.
        ambiguous_fasta (str): Path to ambiguous reads output.
        ambiguous_fasta_lock (FileLock): Lock for ambiguous output.

    Returns:
        tuple | None: (match_type_counts, cell_id_counts, cumulative_barcodes_stats)
        or None when no valid reads.
    """
    reads_in_chunk = len(reads)
    logging.info(f"Post-processing {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

    n_jobs_extract = min(16, reads_in_chunk)
    chunk_contiguous_annotated_sequences = extract_annotated_full_length_seqs(
        reads, predictions, config.model_path_w_CRF,
        actual_lengths, label_binarizer, seq_order,
        barcodes, n_jobs_extract
    )

    # Load whitelists and known sequences for edit distance calculation
    segment_whitelists, known_sequences = {}, {}
    if config.include_edit_distances:
        segment_whitelists, known_sequences = get_or_load_whitelists_and_sequences(
            seq_orders_path=config.seq_orders_path,
            model_name=config.model_name,
            whitelist_df=whitelist_df,
            metadata_path=config.metadata_path,
            sample_id=config.sample_id,
            whitelist_paths=config.whitelist_paths
        )

    # Build DataFrame with all annotations and edit distances
    chunk_df = pd.DataFrame.from_records(
        (
            {
                'ReadName': original_read_names[i],
                'read_length': annotated_read['read_length'],
                'read': annotated_read['read'],
                **build_position_cols(annotated_read),
                **build_sequence_cols(annotated_read, barcodes),
                **(
                    get_all_edit_dist_cols(
                        annotated_read, segment_whitelists, known_sequences, barcodes
                    ) if config.include_edit_distances else {}
                ),
                'base_qualities': base_qualities[i] if config.output_fmt == "fastq" else None,
                'architecture': annotated_read['architecture'],
                'reason': annotated_read['reason'],
                'orientation': annotated_read['orientation']
            }
            for i, annotated_read in enumerate(chunk_contiguous_annotated_sequences)
        )
    )

    # Truncate long sequences to keep file sizes manageable
    for col in chunk_df.columns:
        if '_Sequences' in col:
            chunk_df[col] = chunk_df[col].apply(
                lambda x: x[:MAX_SEQ_DISPLAY_LENGTH] if isinstance(x, str) and len(x) > MAX_SEQ_DISPLAY_LENGTH else x
            )

    # Filter out invalid reads
    invalid_reads_df = chunk_df[chunk_df['architecture'] == 'invalid']
    valid_reads_df = chunk_df[chunk_df['architecture'] != 'invalid']

    if config.model_type == "HYB" and config.pass_num == 1:
        tmp_invalid_dir = os.path.join(output_dir, "tmp_invalid_reads")
        os.makedirs(tmp_invalid_dir, exist_ok=True)

        tmp_invalid_df = pl.DataFrame({
            'ReadName': invalid_reads_df['ReadName'],
            'read': invalid_reads_df['read'],
            'read_length': invalid_reads_df['read_length']
        })

        tmp_path = f'{tmp_invalid_dir}/{bin_name}.tsv'
        lock_path = f"{tmp_path}.lock"

        if not os.path.exists(lock_path):
            with open(lock_path, 'w') as lock_file:
                lock_file.write('')

        with FileLock(lock_path):
            write_header = not os.path.exists(tmp_path)
            with open(tmp_path, 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                if write_header:
                    writer.writerow(tmp_invalid_df.columns)
                writer.writerows(tmp_invalid_df.rows())

    else:
        if not invalid_reads_df.empty:
            with invalid_file_lock:
                add_header = not os.path.exists(invalid_output_file) or os.path.getsize(invalid_output_file) == 0
                invalid_reads_df.to_csv(invalid_output_file, sep='\t', index=False, mode='a', header=add_header, na_rep='NA')

    # Process valid reads for barcodes
    column_mapping = {}
    for barcode in barcodes:
        column_mapping[barcode] = barcode

    # Process barcodes in parallel
    if not valid_reads_df.empty:
        # Identify columns to preserve after demultiplexing
        # bc_n_demultiplex returns a new DataFrame that only includes specific columns
        # We need to preserve edit distances and segment positions (Starts/Ends/Sequences)
        edit_dist_cols = []
        if config.include_edit_distances:
            edit_dist_cols = [col for col in valid_reads_df.columns
                             if col.endswith('_edit_distance') or col.endswith('_match_orientation')]

        # Identify all segment position and sequence columns (for fixed sequences like p7, RP2, etc.)
        segment_cols = [col for col in valid_reads_df.columns
                       if col.endswith('_Starts') or col.endswith('_Ends') or col.endswith('_Sequences')]
        if not config.include_sequences_in_valid_output:
            segment_cols = [col for col in segment_cols if not col.endswith('_Sequences')]

        # Remove barcode and commonly preserved columns (those already in bc_n_demultiplex output)
        preserved_in_demux = ['cDNA_Starts', 'cDNA_Ends', 'UMI_Starts', 'UMI_Ends',
                             'random_s_Starts', 'random_s_Ends', 'random_e_Starts', 'random_e_Ends',
                             'polyA_Starts', 'polyA_Ends']
        # Also exclude barcode segments since they're handled by bc_n_demultiplex
        for barcode in barcodes:
            preserved_in_demux.extend([f'{barcode}_Starts', f'{barcode}_Ends', f'{barcode}_Sequences'])

        segment_cols = [col for col in segment_cols if col not in preserved_in_demux]

        # Add essential columns that must be preserved (read sequence, metadata, etc.)
        essential_cols = []
        if 'read' in valid_reads_df.columns and config.include_sequences_in_valid_output:
            essential_cols.append('read')
        if 'read_length' in valid_reads_df.columns:
            essential_cols.append('read_length')
        if 'base_qualities' in valid_reads_df.columns and config.include_sequences_in_valid_output:
            essential_cols.append('base_qualities')
        if 'architecture' in valid_reads_df.columns:
            essential_cols.append('architecture')
        if 'reason' in valid_reads_df.columns:
            essential_cols.append('reason')
        if 'orientation' in valid_reads_df.columns:
            essential_cols.append('orientation')

        # Combine all columns to preserve
        cols_to_preserve = edit_dist_cols + segment_cols + essential_cols

        corrected_df, match_type_counts, cell_id_counts = bc_n_demultiplex(
            valid_reads_df, strand,
            list(column_mapping.keys()),
            whitelist_dict, whitelist_df, config.threshold,
            output_dir, config.output_fmt, demuxed_fasta,
            demuxed_fasta_lock, ambiguous_fasta,
            ambiguous_fasta_lock, config.n_jobs
        )

        # Filter out columns that already exist in corrected_df to avoid duplicates
        # bc_n_demultiplex already returns columns like read_length, architecture, reason, orientation
        if cols_to_preserve:
            existing_cols = set(corrected_df.columns)
            cols_to_preserve = [col for col in cols_to_preserve if col not in existing_cols]

            # Only merge if there are actually columns to preserve
            if cols_to_preserve:
                # Merge on ReadName to preserve edit distance, segment position, and essential columns
                preserved_df = valid_reads_df[['ReadName'] + cols_to_preserve]
                corrected_df = corrected_df.merge(preserved_df, on='ReadName', how='left')
                logger.info(f"Preserved {len(cols_to_preserve)} columns in valid reads output")
            else:
                logger.info("No additional columns to preserve - all needed columns already in bc_n_demultiplex output")

        # Compute barcode stats
        for barcode in list(column_mapping.keys()):
            count_column = f'corrected_{barcode}_counts_with_min_dist'
            min_dist_column = f'corrected_{barcode}_min_dist'

            # Update count stats
            chunk_count_data = corrected_df[count_column].value_counts()
            for key, value in chunk_count_data.items():
                cumulative_barcodes_stats[barcode]['count_data'][key] = (
                    cumulative_barcodes_stats[barcode]['count_data'].get(key, 0) + value
                )

            # Update min distance stats
            chunk_min_dist_data = corrected_df[min_dist_column].value_counts()
            for key, value in chunk_min_dist_data.items():
                cumulative_barcodes_stats[barcode]['min_dist_data'][key] = (
                    cumulative_barcodes_stats[barcode]['min_dist_data'].get(key, 0) + value
                )

        # Save valid reads with all edit distance columns
        with valid_file_lock:
            add_header = not os.path.exists(valid_output_file) or os.path.getsize(valid_output_file) == 0
            corrected_df.to_csv(valid_output_file, sep='\t', index=False, mode='a', header=add_header, na_rep='NA')

        logging.info(f"Post-processed {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

        return match_type_counts, cell_id_counts, cumulative_barcodes_stats

    for local_df in ["chunk_df", "corrected_df", "invalid_reads_df", "valid_reads_df"]:
        if local_df:
            del local_df

    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()


def post_process_reads(reads, read_names, strand, output_fmt,
                       base_qualities, model_type, pass_num,
                       model_path_w_CRF, predictions, label_binarizer,
                       cumulative_barcodes_stats, read_lengths,
                       seq_order, add_header, bin_name, chunk_idx, output_dir,
                       invalid_output_file, invalid_file_lock,
                       valid_output_file, valid_file_lock, barcodes,
                       whitelist_df, whitelist_dict, threshold,
                       checkpoint_file, chunk_start, match_type_counter,
                       cell_id_counter, demuxed_fasta, demuxed_fasta_lock,
                       ambiguous_fasta, ambiguous_fasta_lock, njobs,
                       seq_orders_path=None, model_name=None,
                       metadata_path=None, sample_id=None,
                       include_edit_distances=False,
                       include_sequences_in_valid_output=False,
                       whitelist_paths=None):
    """
    Wrapper to construct config and run chunk-level post-processing.

    Args mirror process_full_length_reads_in_chunks_and_save with additional:
        model_type (str): "REG"/"CRF"/"HYB".
        pass_num (int): Current pass number.
        threshold (int): Barcode correction threshold.
        checkpoint_file (str): Path to checkpoint file.
        chunk_start (int): Starting chunk index for checkpointing.
        match_type_counter, cell_id_counter (defaultdict): Shared counters.
        njobs (int): Parallel jobs for demux.
        seq_orders_path, model_name: For loading segment orders.
        whitelist_paths (dict | None): Explicit whitelist segment->path mapping.

    Returns:
        tuple: Updated (cumulative_barcodes_stats, match_type_counter, cell_id_counter)
    """

    # Construct configuration object
    config = AnnotateReadsConfig(
        output_fmt=output_fmt,
        model_type=model_type,
        pass_num=pass_num,
        model_path_w_CRF=model_path_w_CRF,
        threshold=threshold,
        n_jobs=njobs,
        seq_orders_path=seq_orders_path,
        model_name=model_name,
        whitelist_paths=whitelist_paths,
        metadata_path=metadata_path,
        sample_id=sample_id,
        include_edit_distances=include_edit_distances,
        include_sequences_in_valid_output=include_sequences_in_valid_output
    )

    results = process_full_length_reads_in_chunks_and_save(
        config=config,
        reads=reads, original_read_names=read_names,
        strand=strand, base_qualities=base_qualities,
        predictions=predictions, bin_name=bin_name, chunk_idx=chunk_idx,
        label_binarizer=label_binarizer,
        cumulative_barcodes_stats=cumulative_barcodes_stats,
        actual_lengths=read_lengths, seq_order=seq_order,
        add_header=add_header, output_dir=output_dir,
        invalid_output_file=invalid_output_file,
        invalid_file_lock=invalid_file_lock,
        valid_output_file=valid_output_file,
        valid_file_lock=valid_file_lock, barcodes=barcodes,
        whitelist_df=whitelist_df, whitelist_dict=whitelist_dict,
        demuxed_fasta=demuxed_fasta,
        demuxed_fasta_lock=demuxed_fasta_lock,
        ambiguous_fasta=ambiguous_fasta,
        ambiguous_fasta_lock=ambiguous_fasta_lock
    )

    if results is not None:
        match_type_counts, cell_id_counts, cumulative_barcodes_stats = results

        for key, value in match_type_counts.items():
            match_type_counter[key] += value
        for key, value in cell_id_counts.items():
            cell_id_counter[key] += value

    save_checkpoint(checkpoint_file, bin_name, chunk_start)

    gc.collect()

    return cumulative_barcodes_stats, match_type_counter, cell_id_counter


def filtering_reason_stats(reason_counter_by_bin, output_dir):
    """
    Save raw and normalized filtering reason counts by bin.

    Args:
        reason_counter_by_bin (dict): {bin_name: {reason: count}} structure.
        output_dir (str): Directory to write TSV outputs.
    """

    raw_counts_df = pd.DataFrame.from_dict(reason_counter_by_bin, orient='index').fillna(0).T
    total_reads = raw_counts_df.sum(axis=0)
    normalized_data = raw_counts_df.div(total_reads, axis=1)

    raw_counts_df.to_csv(f"{output_dir}/filtered_raw_counts_by_bins.tsv", sep='\t')
    normalized_data.to_csv(f"{output_dir}/filtered_normalized_fractions_by_bins.tsv", sep='\t')

    print(f"Saved raw counts to {output_dir}/filtered_raw_counts_by_bins.tsv")
    print(f"Saved normalized fractions to {output_dir}/filtered_normalized_fractions_by_bins.tsv")


def plot_read_n_cDNA_lengths(output_dir):
    """
    Plot read length and cDNA length distributions from annotations_valid.parquet.

    Args:
        output_dir (str): Directory containing annotations_valid.parquet and destination for plots.
    """
    df = pl.read_parquet(f"{output_dir}/annotations_valid.parquet",
                         columns=["read_length", "cDNA_length"])
    read_lengths = []
    cDNA_lengths = []

    read_lengths.extend(df["read_length"].to_list())
    read_lengths = np.array(read_lengths, dtype=int)

    cDNA_lengths.extend(df["cDNA_length"].to_list())
    cDNA_lengths = np.array(cDNA_lengths, dtype=int)

    log_read_lengths = np.log10(read_lengths[read_lengths > 0])
    log_cDNA_lengths = np.log10(cDNA_lengths[cDNA_lengths > 0])

    with PdfPages(f"{output_dir}/plots/cDNA_len_distr.pdf") as pdf:
        if len(log_read_lengths[log_read_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_read_lengths[log_read_lengths > 0],
                     bins=100, color='blue', edgecolor='black')
            plt.title('Read Length Distribution (Log Scale)')
            plt.xlabel('Log10(Read Length)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        if len(log_cDNA_lengths[log_cDNA_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_cDNA_lengths[log_cDNA_lengths > 0],
                     bins=100, color='blue', edgecolor='black')
            plt.title('cDNA Length Distribution (Log Scale)')
            plt.xlabel('Log10(cDNA Length)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
