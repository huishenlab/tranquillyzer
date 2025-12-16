import gc
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# ======================= collapse labels into order ======================= #


def collapse_labels(arr, read_length):
    read = arr[0:read_length]
    collapsed_array = []
    count_dict = {}
    indices_dict = {}
    prev = None
    start_index = 0

    for i, element in enumerate(read):
        if element != prev:
            if prev is not None:
                collapsed_array.append(prev)
                count_dict[prev] = count_dict.get(prev, 0) + 1
                indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, i)]
            prev = element
            start_index = i

    if prev is not None:
        collapsed_array.append(prev)
        count_dict[prev] = count_dict.get(prev, 0) + 1
        indices_dict[prev] = indices_dict.get(prev, []) + [(start_index, len(read))]

    return collapsed_array, count_dict, indices_dict

# ============ check if elements are in expected order ============ #


def flexible_sliding_match(array, pattern):
    """
    Find full pattern matches allowing 'cDNA' at the start, end, and between concatenated patterns.
    Matches must be exact — 'cDNA' is NOT allowed within a single match.
    """
    # Trim leading and trailing cDNA
    start = 0
    while start < len(array) and array[start] == "cDNA":
        start += 1
    end = len(array)
    while end > start and array[end - 1] == "cDNA":
        end -= 1

    core_array = array[start:end]

    matches = []
    i = 0
    while i <= len(core_array) - len(pattern):
        window = core_array[i:i + len(pattern)]
        if window == pattern:
            # Match found
            matches.append((start + i, start + i + len(pattern) - 1))
            i += len(pattern)
            # Skip trailing cDNA if present before next pattern
            while i < len(core_array) and core_array[i] == "cDNA":
                i += 1
        else:
            i += 1

    return matches


def check_order(collapsed_array, count_dict, expected_order):
    polyA = "polyA" if "polyA" in expected_order else "polyT"
    expected_order_wo_polyA = [x for x in expected_order if x != polyA]

    match_sets = [
        ("+", expected_order),
        ("+", expected_order_wo_polyA),
        ("-", expected_order[::-1]),
        ("-", expected_order_wo_polyA[::-1])
    ]

    all_orientations = {}
    first_orientation = ""
    match_regions = []

    for orientation, pattern in match_sets:
        matches = flexible_sliding_match(collapsed_array, pattern)
        if matches:
            if not first_orientation:
                first_orientation = orientation
            all_orientations[orientation] = all_orientations.get(orientation, 0) + len(matches)
            match_regions.extend(matches)

    if all_orientations:
        total = sum(all_orientations.values())
        orientation = first_orientation
        breakdown = ", ".join(f"{k}:{v}" for k, v in all_orientations.items())

        matched_idx_range = set()
        for start, end in match_regions:
            matched_idx_range.update(range(start, end + 1))

        unmatched = [label for i, label in enumerate(collapsed_array) if i not in matched_idx_range and label != "cDNA"]
        extra_info = f" — extra segments: [{'_'.join(unmatched)}]" if unmatched else ""

        if total == 1 and not unmatched:
            return True, orientation, "valid"
        else:
            reason = f"concatenated reads x{total} ({breakdown}){extra_info}"
            return False, orientation, reason

    # No match at all — fallback
    reason = "Unexpected pattern: [" + "_".join(collapsed_array) + "]"
    return False, "", reason

# =================== process full-length reads =================== #


def process_full_len_reads(data, barcodes, label_binarizer, model_path_w_CRF):

    read, prediction, read_length, seq_order = data

    if model_path_w_CRF:
        prediction = np.asarray(prediction)
        if prediction.ndim == 1:
            prediction = prediction[np.newaxis, :]
            # decoded_prediction = label_binarizer.inverse_transform(prediction)[0]
        decoded_prediction = label_binarizer.classes_[prediction[0] if prediction.ndim == 2 else prediction]
    else:
        decoded_prediction = label_binarizer.inverse_transform(prediction)

    # read_length = read_length - 1

    decoded_prediction = decoded_prediction[0:read_length]

    collapsed_array, count_dict, indices_dict = collapse_labels(decoded_prediction, read_length)
    order_match, order, reasons = check_order(collapsed_array,
                                              count_dict,
                                              seq_order)

    annotations = {element: {'Starts': [], 'Ends': [], 'Sequences': []} for element in seq_order}
    annotations['random_s'] = {'Starts': [], 'Ends': [], 'Sequences': []}
    annotations['random_e'] = {'Starts': [], 'Ends': [], 'Sequences': []}
    annotations['read'] = read[0:read_length]

    for element in indices_dict:
        for coordinates in indices_dict[element]:
            start, end = coordinates
            annotations[element]['Starts'].append(start)
            annotations[element]['Ends'].append(end)
    if order_match:
        if len(annotations["cDNA"]['Starts']) == 2:
            if order == "+" and collapsed_array[0] == "cDNA":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]
            elif order == "+" and collapsed_array[-1] == "cDNA":
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][1])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][1])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][0]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][0]]
            elif order == "-" and collapsed_array[-1] == "cDNA":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][1])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][1])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][0]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][0]]
            elif order == "-" and collapsed_array[0] == "cDNA":
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]

        if len(annotations["cDNA"]['Starts']) == 3:
            if order == "+":
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][2])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][2])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]
            else:
                annotations['random_e']['Starts'].append(annotations['cDNA']['Starts'][0])
                annotations['random_e']['Ends'].append(annotations['cDNA']['Ends'][0])
                annotations['random_s']['Starts'].append(annotations['cDNA']['Starts'][2])
                annotations['random_s']['Ends'].append(annotations['cDNA']['Ends'][2])
                annotations['cDNA']['Starts'] = [annotations['cDNA']['Starts'][1]]
                annotations['cDNA']['Ends'] = [annotations['cDNA']['Ends'][1]]

    annotations["architecture"] = "valid" if order_match else "invalid"
    annotations['read_length'] = str(read_length)
    annotations['orientation'] = order
    annotations["reason"] = reasons

    # Extract sequences for ALL segments in ALL reads (valid and invalid)
    # This allows inspection of what the model detected, including segment lengths and edit distances
    # Keep full sequences here for accurate edit distance calculation
    # Truncation for display will happen later in the export process
    # Yes, this is extra overhead, but is necessary for now. Can disable in the future.
    for segment in seq_order:
        if segment in annotations and annotations[segment].get('Starts') and annotations[segment].get('Ends'):
            if annotations[segment]['Starts']:  # Check if list is not empty
                start_pos = int(annotations[segment]['Starts'][0])
                end_pos = int(annotations[segment]['Ends'][0])
                full_seq = read[start_pos:end_pos]
                annotations[segment]["Sequences"] = [full_seq]

    return annotations


def extract_annotated_full_length_seqs(new_data, predictions,
                                       model_path_w_CRF,
                                       read_lengths, label_binarizer,
                                       seq_order, barcodes, n_jobs):

    data = [(new_data[i], predictions[i], read_lengths[i], seq_order) for i in range(len(new_data))]

    annotated_data = []

    if n_jobs == 1:
        for i in range(len(data)):
            annotated_data.append(process_full_len_reads(data[i], barcodes,
                                                         label_binarizer,
                                                         model_path_w_CRF))

    elif n_jobs > 1:
        with mp.Pool(processes=n_jobs) as pool:
            annotated_data = pool.starmap(process_full_len_reads, [(d, barcodes, label_binarizer, model_path_w_CRF) for d in data])
            pool.close()

    del data
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()
    return annotated_data
