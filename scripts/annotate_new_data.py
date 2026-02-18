import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import gc
import time
import json
import logging
import contextlib
import numpy as np
from typing import Optional
from numba import njit
import polars as pl
import tensorflow as tf

from scripts.train_new_model import ont_read_annotator
import scripts.available_gpus as available_gpus
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# TODO: Will need to think about how to handle this when users are able to pick
#       which specific GPUs to use. This may need to be set elsewhere, or still
#       set here, but using a different function

from scripts.available_gpus import gpus_to_visible_devices_string
os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_visible_devices_string()

tf.config.experimental.enable_tensor_float_32_execution(False)
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

logger = logging.getLogger(__name__)

# Create a NumPy lookup array (256 elements to handle all ASCII characters)
NUCLEOTIDE_TO_ID = np.zeros(256, dtype=np.int32)
NUCLEOTIDE_TO_ID[ord("A")] = 1
NUCLEOTIDE_TO_ID[ord("C")] = 2
NUCLEOTIDE_TO_ID[ord("G")] = 3
NUCLEOTIDE_TO_ID[ord("T")] = 4
NUCLEOTIDE_TO_ID[ord("N")] = 5  # Default encoding for unknown nucleotides

# Enable memory growth to avoid pre-allocating all GPU memory
if available_gpus.n_gpus() > 0:
    for gpu in available_gpus.get_tensorflow_output():
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.optimizer.set_jit(True)


@njit
def encode_sequence_numba(read, encoded_seq):
    """nucleotide encoding using ASCII lookup."""
    for i in range(len(read)):
        encoded_seq[i] = NUCLEOTIDE_TO_ID[ord(read[i])]
    return encoded_seq


def preprocess_sequences(sequences, max_len):
    """Converts DNA sequences into NumPy integer arrays."""
    # max_len = max(len(seq) for seq in sequences)  # Get max sequence length
    encoded_array = np.zeros((len(sequences), max_len), dtype=np.int32)  # Pre-allocate array

    for i, seq in enumerate(sequences):
        encoded_array[i, : len(seq)] = encode_sequence_numba(seq, encoded_array[i, : len(seq)])

    return encoded_array


# Function to calculate the total number of rows in the Parquet file
def calculate_total_rows(parquet_file):
    df = pl.scan_parquet(parquet_file)
    total_rows = df.collect().shape[0]
    return total_rows


# Modified function to estimate the average read length from the bin name
def estimate_average_read_length_from_bin(bin_name):
    bounds = bin_name.replace("bp", "").split("_")
    lower_bound = int(bounds[0])
    upper_bound = int(bounds[1])
    return (lower_bound + upper_bound) / 2


def num_replicas(strategy=None):
    return getattr(strategy, "num_replicas_in_sync", 1) if strategy else 1


def bytes_from_gb(gb: float) -> int:
    return int(float(gb) * (1024**3))


def parse_gpu_total_gb(user_total_gb: Optional[str], num_gpus: int) -> list[int]:
    """
    Accepts:
      - None -> default 12 GB per GPU
      - "48" -> broadcast 48 GB to all GPUs
      - "48,48,24" -> per-GPU totals (len must match or first is broadcast)
    Returns list of bytes per GPU.
    """
    default_gb = 12.0
    if not user_total_gb:
        return [bytes_from_gb(default_gb) for _ in range(num_gpus)]
    parts = [p.strip() for p in user_total_gb.split(",") if p.strip()]
    if len(parts) == 1:
        gb = float(parts[0])
        return [bytes_from_gb(gb) for _ in range(num_gpus)]
    vals = [float(p) for p in parts]
    if len(vals) != num_gpus:
        # Length mismatch: broadcast the first value
        return [bytes_from_gb(vals[0]) for _ in range(num_gpus)]
    return [bytes_from_gb(v) for v in vals]


def usable_bytes_per_gpu(total_bytes_per_gpu: list[int], safety_margin: float = 0.35) -> list[int]:
    """
    usable ≈ (total - TF_current) * (1 - safety_margin), clamped to >= 0
    """
    out = []
    handles = available_gpus.get_gpu_names_clean()
    for i, total in enumerate(total_bytes_per_gpu):
        try:
            info = tf.config.experimental.get_memory_info(handles[i])  # {'current','peak'}
            current = int(info.get("current", 0))
        except Exception:
            current = 0
        free_like = max(0, total - current)
        usable = int(max(0, free_like * (1.0 - safety_margin)))
        out.append(usable)
    return out


def pick_per_replica_batch_by_tokens(seq_len, target_tokens_per_replica=1_200_000, min_b=1, max_b=8192):
    if seq_len <= 0:
        return min_b
    b = target_tokens_per_replica // int(seq_len)
    return int(max(min_b, min(max_b, b)))


def pick_per_replica_batch_by_conv(
    usable_bytes_per_gpu,
    seq_len,
    conv_filters=256,
    bytes_per_elem=4,  # assume fp32 conv activations/workspace
    min_b=1,
    max_b=8192,
):
    """
    For each GPU: B <= usable / (C * L * bytes_per_elem).
    Returns the minimum across GPUs (most constrained device).
    """
    caps = []
    denom = max(1, int(conv_filters) * int(seq_len) * int(bytes_per_elem))
    for usable in usable_bytes_per_gpu:
        if usable <= 0:
            caps.append(min_b)
        else:
            caps.append(int(max(min_b, min(max_b, usable // denom))))
    return min(caps) if caps else min_b


def choose_global_batch(
    L,
    conv_filters=256,
    strategy=None,
    target_tokens_per_replica=1_200_000,
    min_b=1,
    max_b=8192,
    user_total_gb: Optional[str] = None,
    safety_margin: float = 0.35,
):
    """
    Choose per-replica batch = min( token-based, conv-based across GPUs ),
    then scale by replicas.
    'user_total_gb' is a string like "48" or "48,48,24".
    If None, default 12 GB/GPU.
    """
    n_gpus = available_gpus.n_gpus()
    if n_gpus == 0:
        # CPU-only fallback: use token-based
        per_replica = pick_per_replica_batch_by_tokens(L, target_tokens_per_replica, min_b, max_b)
        return per_replica  # global==per_replica on CPU

    totals = parse_gpu_total_gb(user_total_gb, num_gpus=n_gpus)
    usable = usable_bytes_per_gpu(totals, safety_margin=safety_margin)

    limit_conv = pick_per_replica_batch_by_conv(usable, L, conv_filters, 4, min_b, max_b)
    limit_tok = pick_per_replica_batch_by_tokens(L, target_tokens_per_replica, min_b, max_b)

    per_replica = max(min_b, min(limit_conv, limit_tok))
    replicas = num_replicas(strategy)
    global_b = max(1, per_replica * replicas)
    return global_b


def predict_with_backoff(model, build_dataset_fn, start_batch: int, min_batch: int = 1):
    bs = int(start_batch)
    last_err = None
    while bs >= int(min_batch):
        try:
            ds = build_dataset_fn(bs)
            result = model.predict(ds, verbose=0)
            return result, bs
        except (tf.errors.ResourceExhaustedError, tf.errors.CancelledError, tf.errors.InternalError) as e:
            last_err = e
            logger.warning(f"OOM at batch={bs}. Retrying with smaller batch...")
            K.clear_session()
            gc.collect()
            for dev in available_gpus.get_gpu_names_raw():
                try:
                    tf.config.experimental.reset_memory_stats(dev)
                except Exception:
                    pass
            time.sleep(1.0)
            bs = max(int(min_batch), bs // 2)
    raise RuntimeError("Prediction OOM/cancelled even at batch=1") from last_err


_batch_log_once = set()


def _log_batch_once(bin_name: str, bs: int):
    if bin_name and bin_name not in _batch_log_once:
        logger.info(f"Using batch_size={bs} for bin {bin_name}")
        _batch_log_once.add(bin_name)


def annotate_new_data_parallel(new_encoded_data, model, global_bs, min_batch=1):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if available_gpus.n_gpus() == 0:
        cpu_bs = min(32, max(1, len(new_encoded_data)))  # small & predictable for tests/CI
        return model.predict(new_encoded_data, batch_size=cpu_bs, verbose=0)

    def build_ds(bs: int):
        return (
            tf.data.Dataset.from_tensor_slices(new_encoded_data)
            .batch(int(bs), drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

    preds, _ = predict_with_backoff(model, build_ds, global_bs, min_batch)
    return preds


def build_model(model_path_w_CRF, model_path, conv_filters, num_labels, strategy=None):
    if model_path_w_CRF:
        model_params_json_path = model_path_w_CRF.replace(".h5", "_params.json")
        with open(model_params_json_path) as f:
            raw_params = json.load(f)
        params = {
            k: (
                v.lower() == "true"
                if isinstance(v, str) and v.lower() in ["true", "false"]
                else int(v)
                if isinstance(v, str) and v.isdigit()
                else float(v)
                if isinstance(v, str) and v.replace(".", "", 1).isdigit()
                else v
            )
            for k, v in raw_params.items()
        }
        conv_filters = int(params.get("conv_filters", conv_filters))

        scope = strategy.scope() if strategy else contextlib.nullcontext()
        with scope:
            model = ont_read_annotator(
                vocab_size=params["vocab_size"],
                embedding_dim=params["embedding_dim"],
                num_labels=num_labels,
                conv_layers=params["conv_layers"],
                conv_filters=conv_filters,
                conv_kernel_size=params["conv_kernel_size"],
                lstm_layers=params["lstm_layers"],
                lstm_units=params["lstm_units"],
                bidirectional=params["bidirectional"],
                attention_heads=params["attention_heads"],
                dropout_rate=params["dropout_rate"],
                regularization=params["regularization"],
                learning_rate=params["learning_rate"],
                crf_layer=True,
            )
        _ = model(tf.zeros((1, 512), dtype=tf.int32), 
                  training=False)
        model.load_weights(model_path_w_CRF)
    else:
        scope = strategy.scope() if strategy else contextlib.nullcontext()
        with scope:
            model = load_model(model_path)
    return model


def model_predictions(
    parquet_file,
    chunk_start,
    chunk_size,
    model_path,
    model_path_w_CRF,
    model_type,
    num_labels,
    user_total_gb: Optional[str] = None,  # e.g. "48" or "48,48,24"; None → 12 GB/GPU
    target_tokens_per_replica: int = 1_200_000,  # tune 0.8–1.5M per your GPUs
    safety_margin: float = 0.35,  # keep ~35% VRAM headroom
    min_batch: int = 1,
    max_batch: int = 8192,
):
    total_rows = calculate_total_rows(parquet_file)
    bin_name = os.path.basename(parquet_file).replace(".parquet", "")

    # Estimate the average read length from the bin name and adjust chunk size
    estimated_avg_length = estimate_average_read_length_from_bin(bin_name)
    dynamic_chunk_size = int(chunk_size * (500 / estimated_avg_length))
    dynamic_chunk_size = min(dynamic_chunk_size, 500000)

    scan_df = pl.scan_parquet(parquet_file)
    num_chunks = (total_rows // dynamic_chunk_size) + (1 if total_rows % dynamic_chunk_size > 0 else 0)

    # Build/load model once (CPU: no strategy; GPU: MirroredStrategy)
    n_gpus = available_gpus.n_gpus()
    min_batch = int(n_gpus) if n_gpus > 0 else min_batch
    strategy = tf.distribute.MirroredStrategy() if n_gpus > 1 else None

    conv_filters = 256  # default, will be overwritten if params present    

    max_len = int(int(bin_name.replace("bp", "").split("_")[1]) + 10)
    global_bs = choose_global_batch(
        max_len,
        conv_filters=conv_filters,
        strategy=strategy,
        target_tokens_per_replica=target_tokens_per_replica,
        min_b=min_batch,
        max_b=max_batch,
        user_total_gb=user_total_gb,
        safety_margin=safety_margin,
    )
    model = build_model(model_path_w_CRF, model_path, conv_filters, num_labels, strategy=strategy)

    for chunk_idx in range(chunk_start, num_chunks + 1):
        logger.info(f"Inferring labels for {bin_name}: chunk {chunk_idx}")

        df_chunk = scan_df.slice((chunk_idx - 1) * dynamic_chunk_size, dynamic_chunk_size).collect()
        read_names = df_chunk["ReadName"].to_list()
        reads = df_chunk["read"].to_list()
        read_lengths = df_chunk["read_length"].to_list()
        base_qualities = df_chunk["base_qualities"].to_list() if "base_qualities" in df_chunk.columns else None

        encoded_data = preprocess_sequences(reads, max_len)
        X_new_padded = pad_sequences(encoded_data, padding='post', dtype='int32', maxlen=max_len)

        if global_bs >= 100 and len(reads) >= 100:
            _log_batch_once(bin_name or "", int(global_bs))

            chunk_predictions = annotate_new_data_parallel(
                X_new_padded,
                model,
                global_bs,
                min_batch=min_batch
            )
        else:
            model = build_model(model_path_w_CRF, model_path, conv_filters, num_labels, strategy=None)
            _log_batch_once(bin_name or "", int(global_bs))

            chunk_predictions = annotate_new_data_parallel(
                X_new_padded,
                model,
                global_bs,
                min_batch=min_batch
            )

        del df_chunk, X_new_padded
        gc.collect()
        logger.info(f"Inferred labels for {bin_name}: chunk {chunk_idx}")

        yield (
            parquet_file,
            bin_name,
            chunk_idx,
            chunk_predictions,
            read_names,
            reads,
            read_lengths,
            base_qualities,
        )
