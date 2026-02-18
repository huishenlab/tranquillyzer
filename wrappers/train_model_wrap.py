# Want logger available to all objects in module
import logging

logger = logging.getLogger(__name__)


def load_libs():
    import os
    import gc
    import json
    import time
    import pickle
    import random
    import itertools
    from collections import Counter

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.utils import shuffle

    from scripts.train_new_model import (
        DynamicPaddingDataGenerator,
        ont_read_annotator,
    )
    from scripts.trained_models import seq_orders
    from scripts.simulate_training_data import generate_training_reads
    from scripts.annotate_new_data import (
        annotate_new_data_parallel,
        preprocess_sequences,
    )
    from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs
    from scripts.visualize_annot import save_plots_to_pdf
    from scripts.available_gpus import log_gpus_used

    return (
        os,
        gc,
        json,
        time,
        pickle,
        random,
        itertools,
        Counter,
        np,
        pd,
        tf,
        SeqIO,
        Seq,
        SeqRecord,
        LabelBinarizer,
        shuffle,
        generate_training_reads,
        seq_orders,
        ont_read_annotator,
        DynamicPaddingDataGenerator,
        annotate_new_data_parallel,
        preprocess_sequences,
        extract_annotated_full_length_seqs,
        save_plots_to_pdf,
        log_gpus_used,
    )


def train_model_wrap(
    model_name,
    output_dir,
    param_file,
    training_seq_orders_file,
    num_val_reads,
    mismatch_rate,
    insertion_rate,
    deletion_rate,
    min_cDNA,
    max_cDNA,
    polyT_error_rate,
    max_insertions,
    threads,
    rc,
    transcriptome,
    invalid_fraction,
    gpu_mem,
    target_tokens,
    vram_headroom,
    min_batch_size,
    max_batch_size,
):
    (
        os,
        gc,
        json,
        time,
        pickle,
        random,
        itertools,
        Counter,
        np,
        pd,
        tf,
        SeqIO,
        Seq,
        SeqRecord,
        LabelBinarizer,
        shuffle,
        generate_training_reads,
        seq_orders,
        ont_read_annotator,
        DynamicPaddingDataGenerator,
        annotate_new_data_parallel,
        preprocess_sequences,
        extract_annotated_full_length_seqs,
        save_plots_to_pdf,
        log_gpus_used,
    ) = load_libs()

    # Let user know whether they're running on CPU only or GPU (provided handles if so)
    log_gpus_used()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(base_dir, "..")

    models_dir = os.path.join(base_dir, "models")
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "utils")
    utils_dir = os.path.abspath(utils_dir)

    if param_file is None:
        param_file = f"{utils_dir}/training_params.tsv"
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    if training_seq_orders_file is None:
        training_seq_orders_file = f"{utils_dir}/training_seq_orders.tsv"
    if not os.path.exists(training_seq_orders_file):
        raise FileNotFoundError(f"Seq orders file not found: {training_seq_orders_file}")

    with open(f"{output_dir}/simulated_data/reads.pkl", "rb") as r:
        reads = pickle.load(r)
    with open(f"{output_dir}/simulated_data/labels.pkl", "rb") as labs:
        labels = pickle.load(labs)

    df = pd.read_csv(param_file, sep="\t")

    # Ensure the model exists in the file
    if model_name not in df.columns:
        logger.info(f"{model_name} not found in the parameter file.")
        return
    logger.info(f"Extracting parameters for {model_name}")

    # Convert the model's parameter column to a dictionary of lists
    param_dict = {df.iloc[i, 0]: df.iloc[i][model_name].split(",") for i in range(len(df))}

    # Generate all possible combinations of parameters for this model
    param_combinations = list(itertools.product(*param_dict.values()))
    length_range = (min_cDNA, max_cDNA)
    seq_order, sequences, barcodes, UMIs, strand = seq_orders(training_seq_orders_file, model_name)

    print(f"seq orders: {seq_order}")

    validation_segment_order = ["cDNA"]
    validation_segment_order.extend(seq_order)
    validation_segment_order.append("cDNA")

    validation_segment_pattern = ["RN"]
    validation_segment_pattern.extend(sequences)
    validation_segment_pattern.append("RN")

    if transcriptome:
        logger.info("Loading transcriptome fasta file")
        transcriptome_records = list(SeqIO.parse(transcriptome, "fasta"))
        logger.info("Transcriptome fasta loaded")
    else:
        logger.info("No transcriptome provided. Will generate random transcripts...")
        transcriptome_records = []
        for i in range(num_val_reads):
            length = random.randint(min_cDNA, max_cDNA)
            seq_str = "".join(np.random.choice(list("ATCG")) for _ in range(length))
            record = SeqRecord(
                Seq(seq_str), id=f"random_transcript_{i + 1}", description=f"Synthetic transcript {i + 1}"
            )
            transcriptome_records.append(record)
        logger.info(f"Generated {len(transcriptome_records)} synthetic transcripts")

    validation_reads, validation_labels = generate_training_reads(
        num_val_reads,
        mismatch_rate,
        insertion_rate,
        deletion_rate,
        polyT_error_rate,
        max_insertions,
        validation_segment_order,
        validation_segment_pattern,
        length_range,
        threads,
        rc,
        transcriptome_records,
        invalid_fraction,
    )

    palette = ["red", "blue", "green", "purple", "pink", "cyan", "magenta", "orange", "brown"]
    colors = {"random_s": "black", "random_e": "black", "cDNA": "gray", "polyT": "orange", "polyA": "orange"}

    i = 0
    for element in seq_order:
        if element not in ["random_s", "random_e", "cDNA", "polyT", "polyA"]:
            colors[element] = palette[i % len(palette)]  # Cycle through the palette
            i += 1

    validation_read_names = range(len(validation_reads))
    validation_read_lengths = []

    for validation_read in validation_reads:
        validation_read_lengths.append(len(validation_read))

    for idx, param_set in enumerate(param_combinations):
        model_filename = f"{model_name}_{idx}.h5"
        param_filename = f"{model_name}_{idx}_params.json"

        os.makedirs(f"{output_dir}/{model_name}_{idx}", exist_ok=True)
        params = dict(zip(param_dict.keys(), param_set))

        # Extract model parameters
        batch_size = int(params.get("batch_size", 64))
        train_fraction = float(params.get("train_fraction", 0.80))
        vocab_size = int(params["vocab_size"])
        embedding_dim = int(params["embedding_dim"])
        num_labels = int(len(seq_order))
        conv_layers = int(params["conv_layers"])
        conv_filters = int(params["conv_filters"])
        conv_kernel_size = int(params["conv_kernel_size"])
        lstm_layers = int(params["lstm_layers"])
        lstm_units = int(params["lstm_units"])
        bidirectional = params["bidirectional"].lower() == "true"
        crf_layer = params["crf_layer"].lower() == "true"
        attention_heads = int(params["attention_heads"])
        dropout_rate = float(params["dropout_rate"])
        regularization = float(params["regularization"])
        learning_rate = float(params["learning_rate"])
        epochs = int(params["epochs"])

        logger.info(f"Training {model_filename} with parameters: {params}")

        # Save the parameters used in training
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{model_name}_{idx}/{param_filename}", "w") as param_file:
            json.dump(params, param_file, indent=4)

        # Shuffle data
        reads, labels = shuffle(reads, labels)

        unique_labels = list(set([item for sublist in labels for item in sublist]))
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(unique_labels)

        # Save label binarizer
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{model_name}_{idx}/{model_name}_{idx}_lbl_bin.pkl", "wb") as lb_file:
            pickle.dump(label_binarizer, lb_file)

        # Train-validation split
        split_index = int(len(reads) * train_fraction)
        train_reads = reads[:split_index]
        train_labels = labels[:split_index]
        val_reads = reads[split_index:]
        val_labels = labels[split_index:]

        logger.info(f"Training reads: {len(train_reads)}, Validation reads: {len(val_reads)}")
        logger.info(f"Training Label Distribution: {Counter([label for seq in train_labels for label in seq])}")
        logger.info(f"Validation Label Distribution: {Counter([label for seq in val_labels for label in seq])}")

        # Data generators
        train_gen = DynamicPaddingDataGenerator(train_reads, train_labels, batch_size, label_binarizer)
        val_gen = DynamicPaddingDataGenerator(val_reads, val_labels, batch_size, label_binarizer)

        # Multi-GPU strategy
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Number of devices: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            model = ont_read_annotator(
                vocab_size,
                embedding_dim,
                num_labels,
                conv_layers=conv_layers,
                conv_filters=conv_filters,
                conv_kernel_size=conv_kernel_size,
                lstm_layers=lstm_layers,
                lstm_units=lstm_units,
                bidirectional=bidirectional,
                crf_layer=crf_layer,
                attention_heads=attention_heads,
                dropout_rate=dropout_rate,
                regularization=regularization,
                learning_rate=learning_rate,
            )

        logger.info(f"Training {model_name}_{idx} with parameters: {params}")
        if crf_layer:
            dummy_input = tf.zeros((1, 512), dtype=tf.int32)  # Batch of 1, sequence length 512
            _ = model(dummy_input)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_val_accuracy", factor=0.5, patience=1, min_lr=1e-5, mode="max"
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss_val", patience=1, restore_best_weights=True
            )

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=[early_stopping, reduce_lr],
                workers=0,
                use_multiprocessing=False,
            )
            model.save_weights(f"{output_dir}/{model_name}_{idx}/{model_name}_{idx}.h5")
        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", factor=0.5, patience=1, min_lr=1e-5, mode="max"
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

            history = model.fit(
                train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stopping, reduce_lr]
            )
            model.save(f"{output_dir}/{model_name}_{idx}/{model_filename}")

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f"{output_dir}/{model_name}_{idx}/{model_name}_{idx}_history.tsv", sep="\t", index=False)

        max_read_len = int(max(validation_read_lengths)) + 10

        encoded_data = preprocess_sequences(validation_reads, max_read_len)
        predictions = annotate_new_data_parallel(
            encoded_data, model, max_batch_size, min_batch=min_batch_size
        )
        annotated_reads = extract_annotated_full_length_seqs(
            validation_reads,
            predictions,
            crf_layer,
            validation_read_lengths,
            label_binarizer,
            seq_order,
            barcodes,
            n_jobs=1,
        )
        save_plots_to_pdf(
            validation_reads,
            annotated_reads,
            validation_read_names,
            f"{output_dir}/{model_name}_{idx}/{model_name}_{idx}_val_viz.pdf",
            colors,
            chars_per_line=150,
        )
        gc.collect()
