import logging

import polars as pl

logger = logging.getLogger(__name__)

def load_libs():
    import os
    import sys
    import time
    import resource
    import random
    import warnings
    import pickle
    from sklearn.preprocessing import LabelBinarizer

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings('ignore', category=UserWarning,
                        module='tensorflow')

    from scripts.extract_annotated_seqs import (
        extract_annotated_full_length_seqs,
    )
    from scripts.annotate_new_data import build_model, preprocess_sequences
    from scripts.trained_models import trained_models, seq_orders
    from scripts.annotate_new_data import annotate_new_data_parallel
    from scripts.visualize_annot import save_plots_to_pdf
    from scripts.available_gpus import log_gpus_used

    return (os, sys, time, resource, random, pickle,
            LabelBinarizer, extract_annotated_full_length_seqs,
            build_model, preprocess_sequences, trained_models,
            seq_orders, annotate_new_data_parallel, save_plots_to_pdf, log_gpus_used)


def visualize_wrap(output_dir, output_file, model_name, model_type,
                   seq_order_file, gpu_mem, target_tokens, vram_headroom,
                   min_batch_size, max_batch_size, num_reads, read_names, threads):

    (os, sys, time, resource, random, pickle,
     LabelBinarizer, extract_annotated_full_length_seqs,
     build_model, preprocess_sequences, trained_models,
     seq_orders, annotate_new_data_parallel, save_plots_to_pdf, log_gpus_used) = load_libs()

    # Exit early if bad inputs given
    if not num_reads and not read_names:
        logger.error("You must either provide a value for 'num_reads' or specify 'read_names'.")
        raise ValueError("You must either provide a value for 'num_reads' or specify 'read_names'.")

    start = time.time()

    # Let user know whether they're running on CPU only or GPU (provided handles if so)
    log_gpus_used()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    models_dir = os.path.join(base_dir, "models")
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "utils")
    utils_dir = os.path.abspath(utils_dir)

    if seq_order_file is None:
        seq_order_file = os.path.join(utils_dir, "seq_orders.tsv")

    seq_order, sequences, barcodes, UMIs, strand = seq_orders(
        seq_order_file,
        model_name
    )

    num_labels = len(seq_order)
    model_path_w_CRF = None
    model_path = None

    conv_filters = 256

    if model_type == "REG":
        model_path = os.path.join(models_dir, model_name + ".h5")
        with open(os.path.join(models_dir, model_name + "_lbl_bin.pkl"), "rb") as f:
            label_binarizer = pickle.load(f)
    else:
        model_path_w_CRF = os.path.join(models_dir, model_name + "_w_CRF.h5")
        with open(os.path.join(models_dir,
                               model_name + "_w_CRF_lbl_bin.pkl"),
                               "rb") as f:
            label_binarizer = pickle.load(f)

    try:
        model = build_model(model_path_w_CRF, model_path, conv_filters, num_labels, strategy=None)
    except Exception as e:
        logger.error(f"Error encountered while building model: {e}")
        sys.exit(1)

    palette = ['red', 'blue', 'green', 'purple', 'pink',
               'cyan', 'magenta', 'orange', 'brown']
    colors = {'random_s': 'black', 'random_e': 'black',
              'cDNA': 'gray', 'polyT': 'orange', 'polyA': 'orange'}
    i = 0
    for element in seq_order:
        if element not in ['random_s', 'random_e', 'cDNA', 'polyT', 'polyA']:
            colors[element] = palette[i % len(palette)]  # Cycle through the palette
            i += 1

    # Path to the read_index.parquet
    index_file_path = os.path.join(output_dir,
                                   "full_length_pp_fa/read_index.parquet")

    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    folder_path = os.path.join(output_dir, "full_length_pp_fa")
    pdf_filename = f'{output_dir}/plots/{output_file}.pdf'

    selected_reads = []
    selected_read_names = []
    selected_read_lengths = []

    # If read_names are provided, visualize those specific reads
    if read_names:
        read_names_list = read_names.split(",")
        missing_reads = []

        for read_name in read_names_list:
            parquet_file = load_read_index(index_file_path, read_name)

            if parquet_file:
                parquet_path = os.path.abspath(parquet_file)

                try:
                    # Load the appropriate Parquet file and retrieve the read
                    df = pl.read_parquet(parquet_path).filter(pl.col("ReadName") == read_name)
                    if not df.is_empty():
                        read_seq = df["read"][0]
                        read_length = df["read_length"][0]
                        selected_reads.append(read_seq)
                        selected_read_names.append(read_name)
                        selected_read_lengths.append(read_length)
                except Exception as e:
                    logger.error(f"Error reading {parquet_path}: {e}")
            else:
                missing_reads.append(read_name)

        if missing_reads:
            logger.warning(f"The following reads were not found in the index: {', '.join(missing_reads)}")

    # If num_reads is provided, randomly select num_reads reads from the index
    elif num_reads:
        df_index = pl.read_parquet(index_file_path)
        all_read_names = df_index["ReadName"].to_list()
        selected_read_names = random.sample(all_read_names,
                                            min(num_reads, len(all_read_names)))

        for read_name in selected_read_names:
            parquet_file = load_read_index(index_file_path, read_name)

            if parquet_file:
                parquet_path = os.path.abspath(parquet_file)

                try:
                    df = pl.read_parquet(parquet_path).filter(pl.col("ReadName") == read_name)
                    if not df.is_empty():
                        read_seq = df["read"][0]
                        read_length = df["read_length"][0]
                        selected_reads.append(read_seq)
                        selected_read_lengths.append(read_length)
                except Exception as e:
                    logger.error(f"Error reading {parquet_path}: {e}")

    # Check if there are any selected reads to process
    if not selected_reads:
        logger.warning("No reads were selected. Skipping inference.")
        return

    # Perform annotation and plotting
    encoded_data = preprocess_sequences(selected_reads)
    try:
        predictions = annotate_new_data_parallel(encoded_data, model,
                                                max_batch_size,
                                                min_batch=min_batch_size,
                                                strategy=None,)
    except Exception as e:
        logger.error(f"Encountered an error during annotation: {e}")
        sys.exit(1)

    annotated_reads = extract_annotated_full_length_seqs(
            selected_reads, predictions, model_path_w_CRF,
            selected_read_lengths, label_binarizer, seq_order,
            barcodes, n_jobs=threads,
        )
    save_plots_to_pdf(selected_reads,
                      annotated_reads,
                      selected_read_names,
                      pdf_filename, colors,
                      chars_per_line=150)

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss  # Linux gives KB
    logger.info(f"Peak memory usage: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")

def load_read_index(index_file_path, read_name):
    df = (
        pl.read_parquet(index_file_path)
        .filter(pl.col("ReadName") == read_name)
    )
    if df.is_empty():
        logger.warning(f"Read {read_name} not found in the index.")
        return None
    return df["ParquetFile"][0]
