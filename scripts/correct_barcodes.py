import logging
import gc
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from matplotlib.backends.backend_pdf import PdfPages

from scripts.demultiplex import assign_cell_id

logger = logging.getLogger(__name__)


# Create ASCII lookup table (0-127 only, avoids excess memory usage)
DNA_COMPLEMENT = np.zeros(128, dtype=np.uint8)  # Only map printable ASCII values
DNA_COMPLEMENT[ord("A")] = ord("T")
DNA_COMPLEMENT[ord("T")] = ord("A")
DNA_COMPLEMENT[ord("G")] = ord("C")
DNA_COMPLEMENT[ord("C")] = ord("G")
DNA_COMPLEMENT[ord("N")] = ord("N")  # Keep 'N' unchanged


@njit
def reverse_complement_numba(seq_ascii):
    rev_comp_arr = DNA_COMPLEMENT[seq_ascii[::-1]]  # Reverse & complement
    return rev_comp_arr


def reverse_complement(seq):
    seq_ascii = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)  # Convert string to ASCII uint8
    rev_comp_ascii = reverse_complement_numba(seq_ascii)  # Call optimized function
    return rev_comp_ascii.tobytes().decode("ascii")  # Convert back to string


def correct_barcode(row, column_name, whitelist, threshold):
    observed_barcode = row[column_name]
    reverse_comp_barcode = reverse_complement(observed_barcode)

    # Get distance scores for observed barcode and reverse complement
    candidates = process.extract(observed_barcode, whitelist, scorer=Levenshtein.distance, limit=5)
    candidates_rev = process.extract(reverse_comp_barcode, whitelist, scorer=Levenshtein.distance, limit=5)

    # Combine results and find minimum distance
    all_matches = candidates + candidates_rev
    min_distance = min(match[1] for match in all_matches)

    # Find all closest barcodes with the same minimum distance
    closest_barcodes = [match[0] for match in all_matches if match[1] == min_distance]

    # Handle threshold & multiple matches
    if min_distance > threshold:
        # if len(closest_barcodes) == 1:
        #     return observed_barcode, closest_barcodes[0], min_distance, 1
        return observed_barcode, "NMF", min_distance, len(closest_barcodes)

    return observed_barcode, ",".join(closest_barcodes), min_distance, len(closest_barcodes)


def write_reads_to_fasta(
    batch_reads, output_fmt, demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock
):
    for cell_id, reads in batch_reads.items():
        if cell_id == "ambiguous":
            with ambiguous_fasta_lock:
                fasta_file = open(ambiguous_fasta, "a")
        else:
            with demuxed_fasta_lock:
                fasta_file = open(demuxed_fasta, "a")

        if output_fmt == "fastq":
            for header, sequence, quality in reads:
                fasta_file.write(f"{header}\n{sequence}\n+\n{quality}\n")
        elif output_fmt == "fasta":
            for header, sequence in reads:
                fasta_file.write(f"{header}\n{sequence}\n")

        fasta_file.close()


def process_row(
    row,
    strand,
    barcode_columns,
    whitelist_dict,
    whitelist_df,
    threshold,
    output_dir,
    output_fmt,
    include_barcode_quals_in_header,
    include_polya_in_output,
):
    result = {
        "ReadName": row["ReadName"],
        "read_length": row["read_length"],
        "cDNA_Starts": row["cDNA_Starts"],
        "cDNA_Ends": row["cDNA_Ends"],
        "cDNA_length": int(row["cDNA_Ends"]) - int(row["cDNA_Starts"]),
        "UMI_Starts": row["UMI_Starts"],
        "UMI_Ends": row["UMI_Ends"],
        "random_s_Starts": row["random_s_Starts"],
        "random_s_Ends": row["random_s_Ends"],
        "random_e_Starts": row["random_e_Starts"],
        "random_e_Ends": row["random_e_Ends"],
    }

    if "polyA_Starts" in row and row["polyA_Starts"] != "":
        result["polyA_Starts"] = row["polyA_Starts"]
        result["polyA_Ends"] = row["polyA_Ends"]
        result["polyA_lengths"] = int(row["polyA_Ends"]) - int(row["polyA_Starts"])
    elif "polyT_Starts" in row and row["polyT_Starts"] != "":
        result["polyA_Starts"] = row["polyT_Starts"]
        result["polyA_Ends"] = row["polyT_Ends"]
        result["polyA_lengths"] = int(row["polyT_Ends"]) - int(row["polyT_Starts"])
    else:
        result["polyA_Starts"] = None
        result["polyA_Ends"] = None
        result["polyA_lengths"] = None

    corrected_barcodes = []
    corrected_barcode_seqs = []

    for barcode_column in barcode_columns:
        whitelist = whitelist_dict[barcode_column]
        corrected_barcode, corrected_seq, min_dist, count = correct_barcode(
            row, barcode_column + "_Sequences", whitelist, threshold
        )
        result[f"corrected_{barcode_column}"] = corrected_seq
        result[f"corrected_{barcode_column}_min_dist"] = min_dist
        result[f"corrected_{barcode_column}_counts_with_min_dist"] = count
        result[f"{barcode_column}_Starts"] = row[f"{barcode_column}_Starts"]
        result[f"{barcode_column}_Ends"] = row[f"{barcode_column}_Ends"]
        corrected_barcodes.append(f"{barcode_column}:{corrected_seq}")
        corrected_barcode_seqs.append(corrected_seq)

    corrected_barcodes_str = ";".join(corrected_barcodes)

    orientation = row["orientation"]

    result["architecture"] = row["architecture"]
    result["reason"] = row["reason"]
    result["orientation"] = orientation

    cell_id, local_match_counts, local_cell_counts = assign_cell_id(result, whitelist_df, barcode_columns)
    result["cell_id"] = cell_id

    corrected_barcode_seqs_str = whitelist_dict["cell_ids"][cell_id] if cell_id != "ambiguous" else "ambiguous"

    cDNA_start = int(row["cDNA_Starts"])
    cDNA_end = int(row["cDNA_Ends"])
    cDNA_sequence = row["read"][cDNA_start:cDNA_end]
    umi_sequence = row["read"][int(row["UMI_Starts"]) : int(row["UMI_Ends"])]
    cDNA_quality = row["base_qualities"][cDNA_start:cDNA_end] if output_fmt == "fastq" else None

    polya_seq = None
    polya_qual = None
    if include_polya_in_output and output_fmt in {"fastq", "fasta"}:
        polya_start_token = row.get("polyA_Starts") or row.get("polyT_Starts")
        polya_end_token = row.get("polyA_Ends") or row.get("polyT_Ends")

        try:
            polya_start = (
                int(float(str(polya_start_token).split(",")[0].strip()))
                if polya_start_token not in (None, "", "None")
                else None
            )
            polya_end = (
                int(float(str(polya_end_token).split(",")[0].strip()))
                if polya_end_token not in (None, "", "None")
                else None
            )
            if polya_start is not None and polya_end is not None and polya_end > polya_start:
                polya_seq = row["read"][polya_start:polya_end]
                if output_fmt == "fastq" and row.get("base_qualities"):
                    polya_qual = row["base_qualities"][polya_start:polya_end]
        except (ValueError, TypeError):
            polya_seq = None
            polya_qual = None

    if orientation == "+" and strand == "fwd":
        pass
    elif orientation == "-" and strand == "fwd":
        cDNA_sequence = reverse_complement(cDNA_sequence)
        umi_sequence = reverse_complement(umi_sequence)
        if polya_seq is not None:
            polya_seq = reverse_complement(polya_seq)
    elif orientation == "+" and strand == "rev":
        cDNA_sequence = reverse_complement(cDNA_sequence)
        umi_sequence = reverse_complement(umi_sequence)
        if polya_seq is not None:
            polya_seq = reverse_complement(polya_seq)
    elif orientation == "-" and strand == "rev":
        pass
    else:
        pass

    batch_reads = defaultdict(list)
    barcode_qual_suffix = ""
    if include_barcode_quals_in_header and output_fmt == "fastq":
        base_q = row.get("base_qualities", "")
        qual_tokens = []
        for barcode_column in barcode_columns:
            try:
                start_token = str(row[f"{barcode_column}_Starts"]).split(",")[0].strip()
                end_token = str(row[f"{barcode_column}_Ends"]).split(",")[0].strip()
                if start_token and end_token:
                    start = int(float(start_token))
                    end = int(float(end_token))
                    if base_q and end > start:
                        qual_slice = base_q[start:end]
                        qual_tokens.append(f"{barcode_column}:{qual_slice}")
            except (KeyError, ValueError, TypeError):
                continue

        # Add UMI qualities if present
        try:
            umi_start = int(float(str(row.get("UMI_Starts", "")).split(",")[0].strip()))
            umi_end = int(float(str(row.get("UMI_Ends", "")).split(",")[0].strip()))
            if base_q and umi_end > umi_start:
                umi_slice = base_q[umi_start:umi_end]
                if umi_slice:
                    qual_tokens.append(f"UMI:{umi_slice}")
        except (ValueError, TypeError):
            pass

        if qual_tokens:
            barcode_qual_suffix = f"|BQ:{';'.join(qual_tokens)}"

    sequence_out = cDNA_sequence
    quality_out = cDNA_quality

    if include_polya_in_output and polya_seq is not None:
        if output_fmt == "fastq" and polya_qual is None:
            # Skip appending polyA if qualities are unavailable in FASTQ mode to avoid length mismatch
            pass
        else:
            sequence_out = cDNA_sequence + polya_seq
            if output_fmt == "fastq" and polya_qual is not None:
                quality_out = (quality_out or "") + polya_qual

    if output_fmt == "fasta":
        batch_reads[corrected_barcode_seqs_str].append(
            (
                f">{row['ReadName']}_{corrected_barcode_seqs_str}_{umi_sequence} "
                f"cell_id:{cell_id}|Barcodes:{corrected_barcodes_str}|UMI:{umi_sequence}|orientation:{orientation}",
                sequence_out,
            )
        )
    elif output_fmt == "fastq":
        header = (
            f"@{row['ReadName']}_{corrected_barcode_seqs_str}_{umi_sequence} "
            f"cell_id:{cell_id}|Barcodes:{corrected_barcodes_str}|UMI:{umi_sequence}|orientation:{orientation}"
            f"{barcode_qual_suffix}"
        )
        batch_reads[corrected_barcode_seqs_str].append(
            (
                header,
                sequence_out,
                quality_out if quality_out is not None else row["base_qualities"][cDNA_start:cDNA_end],
            )
        )
    return result, local_match_counts, local_cell_counts, batch_reads


def bc_n_demultiplex(
    chunk,
    strand,
    barcode_columns,
    whitelist_dict,
    whitelist_df,
    threshold,
    output_dir,
    output_fmt,
    demuxed_fasta,
    demuxed_fasta_lock,
    ambiguous_fasta,
    ambiguous_fasta_lock,
    num_cores,
    include_barcode_quals_in_header=False,
    include_polya_in_output=False,
):
    args = [
        (
            row,
            strand,
            barcode_columns,
            whitelist_dict,
            whitelist_df,
            threshold,
            output_dir,
            output_fmt,
            include_barcode_quals_in_header,
            include_polya_in_output,
        )
        for _, row in chunk.iterrows()
    ]
    batch_reads = defaultdict(list)
    results = []

    if num_cores > 1:
        with Pool(num_cores) as pool:
            results = list(
                tqdm(pool.starmap(process_row, args), total=len(chunk), desc="Processing rows", disable=True)
            )

    elif num_cores == 1:
        # Loop through each row sequentially instead of using multiprocessing
        for arg in tqdm(args, total=len(chunk), desc="Processing rows (no parallelism)", disable=True):
            result = process_row(*arg)
            results.append(result)

    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    processed_results = [res[0] for res in results]
    all_match_type_counts = [res[1] for res in results]
    all_cell_id_counts = [res[2] for res in results]

    for res in results:
        for cell_id, reads in res[3].items():
            batch_reads[cell_id].extend(reads)

    write_reads_to_fasta(
        batch_reads, output_fmt, demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock
    )

    match_type_counts = defaultdict(int)
    cell_id_counts = defaultdict(int)

    for match_counts in all_match_type_counts:
        for key, value in match_counts.items():
            match_type_counts[key] += value

    for cell_counts in all_cell_id_counts:
        for key, value in cell_counts.items():
            cell_id_counts[key] += value

    corrected_df = pd.DataFrame(processed_results)

    return corrected_df, match_type_counts, cell_id_counts


def generate_barcodes_stats_pdf(cumulative_barcodes_stats, barcode_columns, pdf_filename="barcode_plots.pdf"):
    with PdfPages(pdf_filename) as pdf:
        for barcode_column in barcode_columns:
            count_data = pd.Series(cumulative_barcodes_stats[barcode_column]["count_data"]).sort_index()
            min_dist_data = pd.Series(cumulative_barcodes_stats[barcode_column]["min_dist_data"]).sort_index()

            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            axs[0].bar(count_data.index, count_data.values, color="skyblue")
            axs[0].set_xlabel("Number of Matches")
            axs[0].set_ylabel("Frequency")
            axs[0].set_title(f"{barcode_column} - Number of Matches")

            axs[1].bar(min_dist_data.index, min_dist_data.values, color="lightgreen")
            axs[1].set_xlabel("Minimum Distance")
            axs[1].set_ylabel("Frequency")
            axs[1].set_title(f"{barcode_column} - Minimum Distance")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
