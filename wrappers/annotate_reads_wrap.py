# Global library imports - used by all functions in module
import logging
import queue
import time

# Share logger across all functions in module
logger = logging.getLogger(__name__)


def load_libs():
    import os
    import gc
    import sys
    import resource
    import pickle
    import pandas as pd
    import multiprocessing as mp
    from multiprocessing import Manager
    from collections import defaultdict
    import psutil
    import polars as pl

    from filelock import FileLock

    from scripts.export_annotations import (
        post_process_reads,
        plot_read_n_cDNA_lengths,
    )
    from scripts.annotate_new_data import (
        calculate_total_rows,
        model_predictions,
        estimate_average_read_length_from_bin,
    )
    from scripts.preprocess_reads import convert_tsv_to_parquet
    from scripts.trained_models import seq_orders
    from scripts.correct_barcodes import generate_barcodes_stats_pdf
    from scripts.demultiplex import generate_demux_stats_pdf
    from scripts.available_gpus import log_gpus_used

    return (
        os,
        gc,
        sys,
        resource,
        pickle,
        mp,
        Manager,
        defaultdict,
        psutil,
        pl,
        FileLock,
        pd,
        model_predictions,
        post_process_reads,
        seq_orders,
        estimate_average_read_length_from_bin,
        calculate_total_rows,
        generate_barcodes_stats_pdf,
        generate_demux_stats_pdf,
        plot_read_n_cDNA_lengths,
        convert_tsv_to_parquet,
        log_gpus_used,
    )


def collect_prediction_stats(
    result_queue, workers, match_type_counter, cell_id_counter, cumulative_barcodes_stats, max_idle_time=60
):
    """Collect results from each model prediction and collate into shared stats"""
    idle_start = None

    while any(worker.is_alive() for worker in workers) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=15)

            # Reset idle start since a new result has been retrieved
            idle_start = None

            if result:
                local_cumulative_stats, local_match_counter, local_cell_counter, _ = result

                # Count how (all matched, majority matched, ambiguous, etc.) barcodes match
                for key, value in local_match_counter.items():
                    match_type_counter[key] = match_type_counter.get(key, 0) + value

                # Number of reads per demuxed cell
                for key, value in local_cell_counter.items():
                    cell_id_counter[key] = cell_id_counter.get(key, 0) + value

                # Stats for barcode matching
                for barcode in local_cumulative_stats.keys():
                    for stat in ["count_data", "min_dist_data"]:
                        for key, value in local_cumulative_stats[barcode][stat].items():
                            cumulative_barcodes_stats[barcode][stat][key] = (
                                cumulative_barcodes_stats[barcode][stat].get(key, 0) + value
                            )
        except queue.Empty:
            # Wait for the queue to get more entries
            if idle_start is None:
                idle_start = time.time()
                logger.info("Result queue idle, waiting for worker results...")
            elif time.time() - idle_start > max_idle_time:
                raise TimeoutError(
                    f"Result queue timed out after no data for {max_idle_time} seconds and no workers finished."
                )


def _empty_results_queue(result_queue, workers, max_idle_time=60):
    """Runs when error encountered during model prediction. Clears queue to allow for clean exit."""
    idle_start = None

    while any(worker.is_alive() for worker in workers) or not result_queue.empty():
        try:
            # We need to get the items in the queue to clear it, but we don't actually want to do anything
            _ = result_queue.get(timeout=15)

            # Reset idle start since we retrieved a new result
            idle_start = None
        except queue.Empty:
            # Wait for the queue to get more entries
            if idle_start is None:
                idle_start = time.time()
                logger.info("Result queue idle during shutdown, waiting for more results")
            elif time.time() - idle_start > max_idle_time:
                # This may be overkill
                raise TimeoutError(
                    f"Result queue timed out during shutdown after no data received for {max_idle_time} seconds with workers still running."
                )

    logger.info("Results queue cleared. Commence shutdown due to error")


def annotate_reads_wrap(
    output_dir,
    whitelist_file,
    output_fmt,
    model_name,
    model_type,
    seq_order_file,
    chunk_size,
    gpu_mem,
    target_tokens,
    vram_headroom,
    min_batch_size,
    max_batch_size,
    bc_lv_threshold,
    threads,
    max_queue_size,
    include_barcode_quals,
    include_polya,
):
    (
        os,
        gc,
        sys,
        resource,
        pickle,
        mp,
        Manager,
        defaultdict,
        psutil,
        pl,
        FileLock,
        pd,
        model_predictions,
        post_process_reads,
        seq_orders,
        estimate_average_read_length_from_bin,
        calculate_total_rows,
        generate_barcodes_stats_pdf,
        generate_demux_stats_pdf,
        plot_read_n_cDNA_lengths,
        convert_tsv_to_parquet,
        log_gpus_used,
    ) = load_libs()

    start = time.time()

    # Let user know whether they're running on CPU only or GPU (provided handles if so)
    log_gpus_used()

    # Read / create / prepare input files and directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    models_dir = os.path.join(base_dir, "models")
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "utils")
    utils_dir = os.path.abspath(utils_dir)

    if seq_order_file is None:
        seq_order_file = os.path.join(utils_dir, "seq_orders.tsv")

    def _available_models(seq_orders_path):
        """Return list of model names defined in seq_orders.tsv (best-effort)."""
        models = []
        try:
            with open(seq_orders_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    model_id = (line.strip().replace("'", "").replace('"', "").split("\t")[0]).strip()
                    if model_id:
                        models.append(model_id)
        except Exception:
            pass
        return models

    # TODO: model_path and model_path_w_CRF can probably be moved into the model if-statement
    #       This may have to wait until post_process_worker has been moved out of this function though
    model_path_w_CRF = None
    model_path = None

    if model_type == "REG" or model_type == "HYB":
        model_path = f"{models_dir}/{model_name}.h5"
        with open(f"{models_dir}/{model_name}_lbl_bin.pkl", "rb") as f:
            label_binarizer = pickle.load(f)

    try:
        seq_order, sequences, barcodes, UMIs, strand = seq_orders(seq_order_file, model_name)
    except Exception as e:
        available = _available_models(seq_order_file)
        suffix = f" Available models: {', '.join(available)}" if available else " No models found in seq_orders file."
        raise ValueError(f"Model '{model_name}' not found in seq_orders file: {seq_order_file}.{suffix}") from e
    if not seq_order:
        available = _available_models(seq_order_file)
        suffix = f" Available models: {', '.join(available)}" if available else " No models found in seq_orders file."
        raise ValueError(f"Model '{model_name}' not found in seq_orders file: {seq_order_file}.{suffix}")
    whitelist_df = pd.read_csv(whitelist_file, sep="\t")
    num_labels = len(seq_order)

    base_folder_path = os.path.join(output_dir, "full_length_pp_fa")

    invalid_output_file = os.path.join(output_dir, "annotations_invalid.tsv")
    valid_output_file = os.path.join(output_dir, "annotations_valid.tsv")

    parquet_files = sorted(
        [
            os.path.join(base_folder_path, f)
            for f in os.listdir(base_folder_path)
            if f.endswith(".parquet") and not f.endswith("read_index.parquet")
        ],
        key=lambda f: estimate_average_read_length_from_bin(os.path.basename(f).replace(".parquet", "")),
    )

    fasta_dir = os.path.join(output_dir, "demuxed_fasta")
    os.makedirs(fasta_dir, exist_ok=True)

    if output_fmt == "fastq":
        logger.info("Selected output format: FASTQ")
        demuxed_fasta = os.path.join(fasta_dir, "demuxed.fastq")
        ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fastq")
    elif output_fmt == "fasta":
        logger.info("Selected output format: FASTA")
        demuxed_fasta = os.path.join(fasta_dir, "demuxed.fasta")
        ambiguous_fasta = os.path.join(fasta_dir, "ambiguous.fasta")

    demuxed_fasta_lock = FileLock(demuxed_fasta + ".lock")
    ambiguous_fasta_lock = FileLock(ambiguous_fasta + ".lock")

    invalid_file_lock = FileLock(invalid_output_file + ".lock")
    valid_file_lock = FileLock(valid_output_file + ".lock")

    # TODO: This entire object could be dropped and use the barcodes list as it comes
    #       from seq_orders. There is only one location where both the key and value are
    #       used from the dictionary. Since its the same value though, we really don't
    #       need to double store these values
    column_mapping = {barcode: barcode for barcode in barcodes}

    whitelist_dict = {
        "cell_ids": {
            idx + 1: "-".join(map(str, row.dropna().unique()))
            for idx, row in whitelist_df[list(column_mapping.values())].iterrows()
        },
        **{
            input_column: whitelist_df[whitelist_column].dropna().unique().tolist()
            for input_column, whitelist_column in column_mapping.items()
        },
    }

    # Create objects shared across processes
    manager = Manager()
    cumulative_barcodes_stats = manager.dict(
        {
            barcode: {
                "count_data": manager.dict(),
                "min_dist_data": manager.dict(),
            }
            for barcode in column_mapping.keys()
        }
    )
    match_type_counter = manager.dict()
    cell_id_counter = manager.dict()

    def post_process_worker(
        task_queue,
        strand,
        output_fmt,
        count,
        header_track,
        result_queue,
        include_barcode_quals,
        include_polya,
    ):
        """Worker function for processing reads and returning results."""
        while True:
            try:
                item = task_queue.get(timeout=10)
                if item is None:
                    break

                parquet_file, bin_name, chunk_idx, predictions, read_names, reads, read_lengths, base_qualities = item

                local_cumulative_stats = {
                    barcode: {"count_data": {}, "min_dist_data": {}} for barcode in column_mapping.keys()
                }
                local_match_counter, local_cell_counter = defaultdict(int), defaultdict(int)

                # FIXME: output_dir comes from outer function
                checkpoint_file = os.path.join(output_dir, "annotation_checkpoint.txt")

                with header_track.get_lock():
                    add_header = header_track.value == 0

                # FIXME: these variables come from outer function:
                # -- model_type
                # -- pass_num
                # -- model_path_w_CRF
                # -- label_binarizer
                # -- seq_order
                # -- output_dir
                # -- invalid_output_file
                # -- invalid_file_lock
                # -- valid_output_file
                # -- valid_file_lock
                # -- barcodes
                # -- whitelist_df
                # -- whitelist_dict
                # -- bc_lv_threshold
                # -- demuxed_fasta
                # -- demuxed_fasta_lock
                # -- ambiguous_fasta
                # -- ambiguous_fasta_lock
                # -- threads
                result = post_process_reads(
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
                    local_cumulative_stats,
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
                    bc_lv_threshold,
                    checkpoint_file,
                    1,
                    local_match_counter,
                    local_cell_counter,
                    demuxed_fasta,
                    demuxed_fasta_lock,
                    ambiguous_fasta,
                    ambiguous_fasta_lock,
                    threads,
                    include_barcode_quals,
                    include_polya,
                )

                if result:
                    local_cumulative_stats, local_match_counter, local_cell_counter = result
                    result_queue.put((local_cumulative_stats, local_match_counter, local_cell_counter, bin_name))
                else:
                    logger.warning(f"No result from post_process_reads in {bin_name}, chunk {chunk_idx}")

                with count.get_lock():
                    count.value += 1

                gc.collect()
            except queue.Empty:
                pass

    num_workers = min(threads, mp.cpu_count() - 1)
    max_queue_size = max(3, num_workers * 2)

    if model_type == "REG" or model_type == "HYB":
        task_queue = mp.Queue(maxsize=max_queue_size)
        result_queue = mp.Queue()
        count = mp.Value("i", 0)
        header_track = mp.Value("i", 0)

        pass_num = 1

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

        workers = [
            mp.Process(
                target=post_process_worker,
                args=(
                    task_queue,
                    strand,
                    output_fmt,
                    count,
                    header_track,
                    result_queue,
                    include_barcode_quals,
                    include_polya,
                ),
            )
            for _ in range(num_workers)
        ]

        logger.info(f"Number of workers = {len(workers)}")

        for worker in workers:
            worker.start()

        # process all the reads with CNN-LSTM model first
        logger.info("Starting first pass with regular model on all the reads")
        try:
            for parquet_file in parquet_files:
                for item in model_predictions(
                    parquet_file,
                    1,
                    chunk_size,
                    model_path,
                    model_path_w_CRF,
                    model_type,
                    num_labels,
                    user_total_gb=gpu_mem,
                    target_tokens_per_replica=target_tokens,
                    safety_margin=vram_headroom,
                    min_batch=min_batch_size,
                    max_batch=max_batch_size,
                ):
                    task_queue.put(item)
                    with header_track.get_lock():
                        header_track.value += 1
        except Exception as e:
            # Wind down queues, close workers when done, print error and exit
            for _ in range(threads):
                task_queue.put(None)

            _empty_results_queue(result_queue, workers)
            for worker in workers:
                worker.join()
                worker.close()

            # TODO: Update error message when checkpoint restart is re-enabled
            logger.error(
                f"Error found while annotating: {e}. Output files may be corrupted - PLEASE DELETE AND START AGAIN. Exiting!"
            )
            sys.exit(1)

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

        for _ in range(threads):
            task_queue.put(None)

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
        collect_prediction_stats(result_queue, workers, match_type_counter, cell_id_counter, cumulative_barcodes_stats)
        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

        logger.info("Finished first pass with regular model on all the reads")

        for worker in workers:
            worker.join()

        logger.info(f"[Memory] RSS: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
        model_path_w_CRF = f"{models_dir}/{model_name}_w_CRF.h5"

        if model_type == "HYB":
            tmp_invalid_dir = os.path.join(output_dir, "tmp_invalid_reads")

            convert_tsv_to_parquet(tmp_invalid_dir, row_group_size=1000000)
            invalid_parquet_files = sorted(
                [
                    os.path.join(tmp_invalid_dir, f)
                    for f in os.listdir(tmp_invalid_dir)
                    if f.endswith(".parquet") and not f.endswith("read_index.parquet")
                ],
                key=lambda f: estimate_average_read_length_from_bin(os.path.basename(f).replace(".parquet", "")),
            )

            with open(f"{models_dir}/{model_name}_w_CRF_lbl_bin.pkl", "rb") as f:
                label_binarizer = pickle.load(f)

            # if model type selcted is HYB, process the failed reads in step 1 with CNN-LSTM-CRF model
            pass_num = 2
            task_queue = mp.Queue(maxsize=max_queue_size)
            result_queue = mp.Queue()

            with count.get_lock():
                count.value = 0

            with header_track.get_lock():
                header_track.value = 0

            workers = [
                mp.Process(
                    target=post_process_worker,
                    args=(
                        task_queue,
                        strand,
                        output_fmt,
                        count,
                        header_track,
                        result_queue,
                        include_barcode_quals,
                        include_polya,
                    ),
                )
                for _ in range(num_workers)
            ]

            for worker in workers:
                worker.start()

            logger.info("Starting second pass with CRF model on invalid reads")
            try:
                for invalid_parquet_file in invalid_parquet_files:
                    if calculate_total_rows(invalid_parquet_file) >= 100:
                        for item in model_predictions(
                            invalid_parquet_file, 1, 
                            chunk_size, model_path,
                            model_path_w_CRF, model_type, 
                            num_labels,
                            user_total_gb=gpu_mem,
                            target_tokens_per_replica=target_tokens,
                            safety_margin=vram_headroom,
                            min_batch=min_batch_size,
                            max_batch=max_batch_size,
                        ):
                            task_queue.put(item)
                            with header_track.get_lock():
                                header_track.value += 1
            except Exception as e:
                # Wind down queues, close workers when done, print error and exit
                for _ in range(threads):
                    task_queue.put(None)

                _empty_results_queue(result_queue, workers)
                for worker in workers:
                    worker.join()
                    worker.close()

                # TODO: Update error message when checkpoint restart is re-enabled
                logger.error(
                    f"Error found while annotating: {e}. Output files may be corrupted - PLEASE DELETE AND START AGAIN. Exiting!"
                )
                sys.exit(1)

            for _ in range(threads):
                task_queue.put(None)

            collect_prediction_stats(
                result_queue, workers, match_type_counter, cell_id_counter, cumulative_barcodes_stats
            )
            logger.info("Finished second pass with CRF model on invalid reads")

            for worker in workers:
                worker.join()

    if model_type == "CRF":
        # process all the reads with CNN-LSTM-CRF model
        model_path_w_CRF = f"{models_dir}/{model_name}_w_CRF.h5"

        with open(f"{models_dir}/{model_name}_w_CRF_lbl_bin.pkl", "rb") as f:
            label_binarizer = pickle.load(f)

        task_queue = mp.Queue(maxsize=max_queue_size)
        result_queue = mp.Queue()
        count = mp.Value("i", 0)
        header_track = mp.Value("i", 0)

        pass_num = 1

        workers = [
            mp.Process(
                target=post_process_worker,
                args=(
                    task_queue,
                    strand,
                    output_fmt,
                    count,
                    header_track,
                    result_queue,
                    include_barcode_quals,
                    include_polya,
                ),
            )
            for _ in range(num_workers)
        ]

        for worker in workers:
            worker.start()

        logger.info("Starting first pass with CRF model on all the reads")
        try:
            for parquet_file in parquet_files:
                for item in model_predictions(
                    parquet_file, 1, chunk_size, None,
                    model_path_w_CRF, model_type,
                    num_labels,
                    user_total_gb=gpu_mem,
                    target_tokens_per_replica=target_tokens,
                    safety_margin=vram_headroom,
                    min_batch=min_batch_size,
                    max_batch=max_batch_size,

                ):
                    task_queue.put(item)
                    with header_track.get_lock():
                        header_track.value += 1
        except Exception as e:
            # Wind down queues, close workers when done, print error and exit
            for _ in range(threads):
                task_queue.put(None)

            _empty_results_queue(result_queue, workers)
            for worker in workers:
                worker.join()
                worker.close()

            # TODO: Update error message when checkpoint restart is re-enabled
            logger.error(
                f"Error found while annotating: {e}. Output files may be corrupted - PLEASE DELETE AND START AGAIN. Exiting!"
            )
            sys.exit(1)

        for _ in range(threads):
            task_queue.put(None)

        collect_prediction_stats(result_queue, workers, match_type_counter, cell_id_counter, cumulative_barcodes_stats)

        logger.info("Finished first pass with CRF model on all the reads")

        for worker in workers:
            worker.join()
            worker.close()

    # Convert shared dictionary to a standard dictionary
    # Each key of the inner dictionary is its own shared dictionary, so convert those as well
    cumulative_barcodes_stats = {
        k: {stat: dict(stat_dict) for stat, stat_dict in inner.items()}
        for k, inner in cumulative_barcodes_stats.items()
    }

    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    logger.info("Generating barcode stats plots")
    generate_barcodes_stats_pdf(
        cumulative_barcodes_stats, list(column_mapping.keys()), pdf_filename=f"{output_dir}/plots/barcode_plots.pdf"
    )
    logger.info("Generated barcode stats plots")

    logger.info("Generating demux stats plots")
    generate_demux_stats_pdf(
        f"{output_dir}/plots/demux_plots.pdf",
        f"{output_dir}/matchType_readCount.tsv",
        f"{output_dir}/cellId_readCount.tsv",
        match_type_counter,
        cell_id_counter,
    )
    logger.info("Generated demux stats plots")

    if os.path.exists(f"{output_dir}/annotations_valid.tsv"):
        with open(f"{output_dir}/annotations_valid.tsv", "r") as f:
            header = f.readline().strip().split("\t")
            dtypes = {col: pl.Utf8 for col in header if col != "read_length"}
            dtypes["read_length"] = pl.Int64

        df = pl.scan_csv(f"{output_dir}/annotations_valid.tsv", separator="\t", dtypes=dtypes)
        annotations_valid_parquet_file = f"{output_dir}/annotations_valid.parquet"
        logger.info("Converting annotations_valid.tsv")
        df.sink_parquet(annotations_valid_parquet_file, compression="snappy", row_group_size=chunk_size)
        logger.info("Converted annotations_valid.tsv to annotations_valid.parquet")
        os.system(f"rm {output_dir}/annotations_valid.tsv")
        os.system(f"rm {output_dir}/annotations_valid.tsv.lock")

        logger.info("Generating valid read length and cDNA length distribution plots")
        plot_read_n_cDNA_lengths(output_dir)
        logger.info("Generated valid read length and cDNA length distribution plots")
        del df
    else:
        logger.warning("annotations_valid.tsv not found. Skipping Parquet conversion.")

    if os.path.exists(f"{output_dir}/annotations_invalid.tsv"):
        logger.info("annotations_invalid.tsv found — proceeding with Parquet conversion")
        with open(f"{output_dir}/annotations_invalid.tsv", "r") as f:
            header = f.readline().strip().split("\t")
        dtypes = {col: pl.Utf8 for col in header if col != "read_length"}
        dtypes["read_length"] = pl.Int64

        df = pl.scan_csv(f"{output_dir}/annotations_invalid.tsv", separator="\t", dtypes=dtypes)
        annotations_invalid_parquet_file = f"{output_dir}/annotations_invalid.parquet"
        logger.info("Converting annotations_invalid.tsv")
        df.sink_parquet(annotations_invalid_parquet_file, compression="snappy", row_group_size=chunk_size)
        logger.info("Converted annotations_invalid.tsv to annotations_invalid.parquet")
        os.system(f"rm {output_dir}/annotations_invalid.tsv")
        os.system(f"rm {output_dir}/annotations_invalid.tsv.lock")
        del df
    else:
        logger.warning("annotations_invalid.tsv not found. Skipping Parquet conversion.")

    if os.path.exists(f"{output_dir}/demuxed_fasta/demuxed.fasta.lock"):
        os.system(f"rm {output_dir}/demuxed_fasta/demuxed.fasta.lock")
    if os.path.exists(f"{output_dir}/demuxed_fasta/ambiguous.fasta.lock"):
        os.system(f"rm {output_dir}/demuxed_fasta/ambiguous.fasta.lock")
    if os.path.exists(f"{output_dir}/demuxed_fasta/demuxed.fastq.lock"):
        os.system(f"rm {output_dir}/demuxed_fasta/demuxed.fastq.lock")
    if os.path.exists(f"{output_dir}/demuxed_fasta/ambiguous.fastq.lock"):
        os.system(f"rm {output_dir}/demuxed_fasta/ambiguous.fastq.lock")

    if model_type == "HYB":
        os.system(f"rm -r {output_dir}/tmp_invalid_reads")

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss  # Linux gives KB
    logger.info(f"Peak memory usage during annotation/barcode correction/demuxing: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
