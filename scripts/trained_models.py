import logging
import os

logger = logging.getLogger(__name__)

def trained_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "..", "models")
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "..", "utils")
    utils_dir = os.path.abspath(utils_dir)

    try:
        # Check if the directory exists
        if not os.path.isdir(models_dir):
            print(f"The directory '{models_dir}' does not exist.")
            return

        print("\n~~~~~~~~~~~~~~~~ CURRENTLY AVAILABLE TRAINED MODELS ~~~~~~~~~~~~~~~~")
        print(
            "\n".join(
                [
                    "-- Sequence Key:",
                    "\tNX ==> unknown sequence of length X",
                    "\tNN ==> unknown sequence of unknown length",
                    "\tA  ==> sequence of A's of unknown length",
                    "\tT  ==> sequence of T's of unknown length",
                    "", # adds in an extra new line between key and models
                ]
            )
        )

        # Iterate over all files in the directory
        for file_name in os.listdir(models_dir):
            # Check if the file has a .h5 extension
            if file_name.endswith('.h5'):
                try:
                    seq_order, sequences, barcodes, UMIs, orientation = seq_orders(os.path.join(utils_dir, "seq_orders.tsv"), file_name[:-3])

                    # Find longest seq_order name
                    longest = max([len(x) for x in seq_order])

                    # Build up elements to be printed
                    print_elements = [
                        f"-- {file_name[:-3]}",
                        "\tlayout (top to bottom) ==> sequence"
                    ]

                    for i in range(len(seq_order)):
                        print_elements.append(f"\t{seq_order[i]:<{longest}} ==> {sequences[i]}")

                    print_elements.append("") # adds in an extra new line between models

                    print("\n".join(print_elements))
                except Exception:
                    print(f"-- {file_name[:-3]}\n\t==> model exists in models/ directory but is undefined in utils/seq_orders.tsv\n")

    except Exception as e:
        print(f"An error occurred: {e}")


def seq_orders(file_path, model):
    """
    Load model configuration from seq_orders.tsv file.

    Args:
        file_path: Path to seq_orders.tsv file
        model: Model name to look up

    Returns:
        Tuple of (sequence_order, sequences, barcodes, UMIs, strand)

    Raises:
        FileNotFoundError: If seq_orders.tsv file doesn't exist
        ValueError: If model not found or entry is malformed
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"ERROR: seq_orders.tsv file not found!\n"
            f"  Expected location: {file_path}\n"
            f"  \n"
            f"  This file is required to define model architectures.\n"
            f"  Please ensure the file exists or specify a custom path with --seq-order-file\n"
            f"{'='*80}"
        )

    # Track all available models for error message
    available_models = []

    # Open the file and read lines
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line by tabs, removing extra quote characters at the same time
            fields = line.replace("'", "").replace("\"", "").split("\t")

            if len(fields) < 1:
                continue

            model_name = fields[0].strip()
            available_models.append(model_name)

            # Check if desired model has been found
            if model_name == model:
                # Validate we have all required fields
                if len(fields) < 5:
                    raise ValueError(
                        f"\n{'='*80}\n"
                        f"ERROR: Malformed entry for model '{model}' in seq_orders.tsv!\n"
                        f"  Location: {file_path}, line {line_num}\n"
                        f"  \n"
                        f"  Expected format (tab-separated):\n"
                        f"    model_name<TAB>segment_order<TAB>sequences<TAB>barcodes<TAB>UMIs<TAB>strand\n"
                        f"  \n"
                        f"  Found {len(fields)} fields, expected at least 5.\n"
                        f"  Fields found: {fields}\n"
                        f"  \n"
                        f"  Missing: {', '.join(['segment_order', 'sequences', 'barcodes', 'UMIs', 'strand'][len(fields)-1:])}\n"
                        f"{'='*80}"
                    )

                try:
                    sequence_order = fields[1].strip().split(',')
                    sequences = fields[2].strip().split(',')
                    barcodes = fields[3].strip().split(',')
                    UMIs = fields[4].strip().split(',')
                    # Strand is optional (default to empty string if not provided)
                    strand = fields[5].strip() if len(fields) > 5 else ""

                    return sequence_order, sequences, barcodes, UMIs, strand

                except IndexError as e:
                    raise ValueError(
                        f"\n{'='*80}\n"
                        f"ERROR: Failed to parse model '{model}' from seq_orders.tsv!\n"
                        f"  Location: {file_path}, line {line_num}\n"
                        f"  \n"
                        f"  Error: {str(e)}\n"
                        f"  Line content: {line}\n"
                        f"{'='*80}"
                    )

    # If we make it here, requested model was not found
    raise ValueError(
        f"\n{'='*80}\n"
        f"ERROR: Model '{model}' not found in seq_orders.tsv!\n"
        f"  File: {file_path}\n"
        f"  \n"
        f"  Available models ({len(available_models)}):\n"
        f"    {', '.join(sorted(available_models))}\n"
        f"  \n"
        f"  To add your model, add a line to seq_orders.tsv with this format:\n"
        f"    {model}<TAB>segment_order<TAB>sequences<TAB>barcodes<TAB>UMIs<TAB>strand\n"
        f"  \n"
        f"  Example:\n"
        f"    {model}<TAB>\"p7,CBC,cDNA,p5\"<TAB>\"ACGT,N16,NN,TGCA\"<TAB>CBC<TAB><TAB>fwd\n"
        f"{'='*80}"
    )


