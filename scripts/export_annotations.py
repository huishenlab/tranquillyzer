import os
import gc
import csv
import logging
import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from filelock import FileLock

from scripts.correct_barcodes import bc_n_demultiplex
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs

logger = logging.getLogger(__name__)


# Function to save the current state (bin_name and chunk) for checkpointing
def save_checkpoint(checkpoint_file, bin_name, chunk):
    with open(checkpoint_file, "w") as f:
        f.write(f"{bin_name},{chunk}")


# Function to load the last checkpoint
def load_checkpoint(checkpoint_file, start_bin):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            bin_name, chunk = f.readline().strip().split(",")
        return bin_name, int(chunk)
    return start_bin, 1  # If no checkpoint, start from the first chunk


def process_full_length_reads_in_chunks_and_save(
    reads,
    original_read_names,
    strand,
    output_fmt,
    base_qualities,
    model_type,
    pass_num,
    model_path_w_CRF,
    predictions,
    bin_name,
    chunk_idx,
    label_binarizer,
    cumulative_barcodes_stats,
    actual_lengths,
    seq_order,
    add_header,
    output_dir,
    invalid_output_file,
    invalid_file_lock,
    valid_output_file,
    valid_file_lock,
    barcodes,
    whitelist_df,
    whitelist_dict,
    demuxed_fasta,
    demuxed_fasta_lock,
    ambiguous_fasta,
    ambiguous_fasta_lock,
    threshold,
    n_jobs,
    include_barcode_quals,
    include_polya,
):
    reads_in_chunk = len(reads)

    logger.info(f"Post-processing {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

    n_jobs_extract = min(16, reads_in_chunk)
    chunk_contiguous_annotated_sequences = extract_annotated_full_length_seqs(
        reads, predictions, model_path_w_CRF, actual_lengths, label_binarizer, seq_order, barcodes, n_jobs_extract
    )

    chunk_df = pd.DataFrame.from_records(
        (
            {
                "ReadName": original_read_names[i],
                "read_length": annotated_read["read_length"],
                "read": annotated_read["read"],
                **{
                    f"{label}_Starts": ", ".join(map(str, annotations["Starts"]))
                    for label, annotations in annotated_read.items()
                    if label not in {"architecture", "reason"} and "Starts" in annotations
                },
                **{
                    f"{label}_Ends": ", ".join(map(str, annotations["Ends"]))
                    for label, annotations in annotated_read.items()
                    if label not in {"architecture", "reason"} and "Ends" in annotations
                },
                **{
                    f"{label}_Sequences": ", ".join(map(str, annotated_read[label]["Sequences"]))
                    for label in barcodes
                    if label in annotated_read and "Sequences" in annotated_read[label]
                },
                "base_qualities": base_qualities[i] if output_fmt == "fastq" else None,
                "architecture": annotated_read["architecture"],
                "reason": annotated_read["reason"],
                "orientation": annotated_read["orientation"],
            }
            for i, annotated_read in enumerate(chunk_contiguous_annotated_sequences)
        )
    )
    # Filter out invalid reads
    invalid_reads_df = chunk_df[chunk_df["architecture"] == "invalid"]
    valid_reads_df = chunk_df[chunk_df["architecture"] != "invalid"]

    if model_type == "HYB" and pass_num == 1:
        tmp_invalid_dir = os.path.join(output_dir, "tmp_invalid_reads")
        os.makedirs(tmp_invalid_dir, exist_ok=True)
        
        tmp_invalid_payload = {
            "ReadName": invalid_reads_df["ReadName"],
            "read": invalid_reads_df["read"],
            "read_length": invalid_reads_df["read_length"],
        }
        if output_fmt == "fastq":
            tmp_invalid_payload["base_qualities"] = invalid_reads_df["base_qualities"]
        
        tmp_invalid_df = pl.DataFrame(tmp_invalid_payload)

        tmp_path = f"{tmp_invalid_dir}/{bin_name}.tsv"
        lock_path = f"{tmp_path}.lock"

        if not os.path.exists(lock_path):
            with open(lock_path, "w") as lock_file:
                lock_file.write("")  # create the lock file

        with FileLock(lock_path):
            write_header = not os.path.exists(tmp_path)
            with open(tmp_path, "a", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                if write_header:
                    writer.writerow(tmp_invalid_df.columns)
                writer.writerows(tmp_invalid_df.rows())

    else:
        if not invalid_reads_df.empty:
            with invalid_file_lock:
                add_header = not os.path.exists(invalid_output_file) or os.path.getsize(invalid_output_file) == 0
                invalid_reads_df.to_csv(invalid_output_file, sep="\t", index=False, mode="a", header=add_header)
        # del invalid_reads_df

    # Process valid reads for barcodes
    column_mapping = {}

    for barcode in barcodes:
        column_mapping[barcode] = barcode

    # Process barcodes in parallel
    if not valid_reads_df.empty:
        # logging.info("Correcting barcode and demuliplexing valid reads")
        corrected_df, match_type_counts, cell_id_counts = bc_n_demultiplex(
            valid_reads_df,
            strand,
            list(column_mapping.keys()),
            whitelist_dict,
            whitelist_df,
            threshold,
            output_dir,
            output_fmt,
            demuxed_fasta,
            demuxed_fasta_lock,
            ambiguous_fasta,
            ambiguous_fasta_lock,
            n_jobs,
            include_barcode_quals,
            include_polya,
        )

        for barcode in list(column_mapping.keys()):
            count_column = f"corrected_{barcode}_counts_with_min_dist"
            min_dist_column = f"corrected_{barcode}_min_dist"

            # Update count stats
            chunk_count_data = corrected_df[count_column].value_counts()
            for key, value in chunk_count_data.items():
                cumulative_barcodes_stats[barcode]["count_data"][key] = (
                    cumulative_barcodes_stats[barcode]["count_data"].get(key, 0) + value
                )

            # Update min distance stats
            chunk_min_dist_data = corrected_df[min_dist_column].value_counts()
            for key, value in chunk_min_dist_data.items():
                cumulative_barcodes_stats[barcode]["min_dist_data"][key] = (
                    cumulative_barcodes_stats[barcode]["min_dist_data"].get(key, 0) + value
                )

        with valid_file_lock:  # FileLock ensures only one process writes at a time
            add_header = not os.path.exists(valid_output_file) or os.path.getsize(valid_output_file) == 0
            corrected_df.to_csv(valid_output_file, sep="\t", index=False, mode="a", header=add_header)

        logger.info(f"Post-processed {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

        return match_type_counts, cell_id_counts, cumulative_barcodes_stats

    for local_df in ["chunk_df", "corrected_df", "invalid_reads_df", "valid_reads_df"]:
        if local_df:
            del local_df

    # Clean up after each chunk to free memory
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()


def post_process_reads(
    reads,
    read_names,
    strand,
    output_fmt,
    base_qualities,
    model_type,
    pass_num,
    model_path_w_CRF,
    predictions,
    label_binarizer,
    cumulative_barcodes_stats,
    read_lengths,
    seq_order,
    add_header,
    bin_name,
    chunk_idx,
    output_dir,
    invalid_output_file,
    invalid_file_lock,
    valid_output_file,
    valid_file_lock,
    barcodes,
    whitelist_df,
    whitelist_dict,
    threshold,
    checkpoint_file,
    chunk_start,
    match_type_counter,
    cell_id_counter,
    demuxed_fasta,
    demuxed_fasta_lock,
    ambiguous_fasta,
    ambiguous_fasta_lock,
    njobs,
    include_barcode_quals,
    include_polya,
):
    results = process_full_length_reads_in_chunks_and_save(
        reads,
        read_names,
        strand,
        output_fmt,
        base_qualities,
        model_type,
        pass_num,
        model_path_w_CRF,
        predictions,
        bin_name,
        chunk_idx,
        label_binarizer,
        cumulative_barcodes_stats,
        read_lengths,
        seq_order,
        add_header,
        output_dir,
        invalid_output_file,
        invalid_file_lock,
        valid_output_file,
        valid_file_lock,
        barcodes,
        whitelist_df,
        whitelist_dict,
        demuxed_fasta,
        demuxed_fasta_lock,
        ambiguous_fasta,
        ambiguous_fasta_lock,
        threshold,
        njobs,
        include_barcode_quals,
        include_polya,
    )

    if results is not None:
        match_type_counts, cell_id_counts, cumulative_barcodes_stats = results

        for key, value in match_type_counts.items():
            match_type_counter[key] += value
        for key, value in cell_id_counts.items():
            cell_id_counter[key] += value

    save_checkpoint(checkpoint_file, bin_name, chunk_start)

    gc.collect()  # Clean up memory after processing each chunk

    return cumulative_barcodes_stats, match_type_counter, cell_id_counter


def filtering_reason_stats(reason_counter_by_bin, output_dir):
    # Convert dictionary to DataFrame (Bins as Columns, Reasons as Rows)
    raw_counts_df = pd.DataFrame.from_dict(reason_counter_by_bin, orient="index").fillna(0).T

    # Compute total reads per bin
    total_reads = raw_counts_df.sum(axis=0)

    # Normalize each column (fraction per bin)
    normalized_data = raw_counts_df.div(total_reads, axis=1)

    # Save both raw counts and normalized fractions
    raw_counts_df.to_csv(f"{output_dir}/filtered_raw_counts_by_bins.tsv", sep="\t")
    normalized_data.to_csv(f"{output_dir}/filtered_normalized_fractions_by_bins.tsv", sep="\t")

    print(f"Saved raw counts to {output_dir}/filtered_raw_counts_by_bins.tsv")
    print(f"Saved normalized fractions to {output_dir}/filtered_normalized_fractions_by_bins.tsv")


def plot_read_n_cDNA_lengths(output_dir):
    
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    df = pl.read_parquet(f"{output_dir}/annotations_valid.parquet", columns=["read_length", "cDNA_length"])
    read_lengths = []
    cDNA_lengths = []

    read_lengths.extend(df["read_length"].to_list())
    read_lengths = np.array(read_lengths, dtype=int)

    cDNA_lengths.extend(df["cDNA_length"].to_list())
    cDNA_lengths = np.array(cDNA_lengths, dtype=int)

    log_read_lengths = np.log10(read_lengths[read_lengths > 0])
    log_cDNA_lengths = np.log10(cDNA_lengths[cDNA_lengths > 0])

    with PdfPages(f"{output_dir}/plots/cDNA_len_distr.pdf") as pdf:
        # valid read length distribution
        if len(log_read_lengths[log_read_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_read_lengths[log_read_lengths > 0], bins=100, color="blue", edgecolor="black")
            plt.title("Read Length Distribution (Log Scale)")
            plt.xlabel("Log10(Read Length)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # cDNA length distribution
        if len(log_cDNA_lengths[log_cDNA_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_cDNA_lengths[log_cDNA_lengths > 0], bins=100, color="blue", edgecolor="black")
            plt.title("cDNA Length Distribution (Log Scale)")
            plt.xlabel("Log10(cDNA Length)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
