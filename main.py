import typer
from typing_extensions import Annotated
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import logging

# Define logging for entire program here
def setup_logger():
    """
    Setup logging for tranquillyzer. Allows for consistent formatting across all
    modules. By putting this in a function, we can also encapsulate all setup in
    one location in case we want to prettify the output. All modules should run

    import logging
    logger = logging.getLogger(__name__)

    at the top of its file to become children of this main logger
    """
    FORMAT = '{asctime} - {name:<35} - {levelname:<7} - {message}'
    logging.basicConfig(format=FORMAT, style='{', level=logging.INFO)

    return logging.getLogger(__name__)

logger = setup_logger()

# =========================
# versioning and app setup
# =========================


try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore[assignment]


def get_version() -> str:
    try:
        return version("tranquillyzer")
    except PackageNotFoundError:
        return "unknown"


app = typer.Typer(rich_markup_mode="rich")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tranquillyzer {get_version()}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Tranquillyzer command-line interface."""
    # Print version banner for all invocations
    typer.echo(f"\nTranquillyzer:v{get_version()}")

    # If invoked without a subcommand → show top-level help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

# =========================
# available trained models
# =========================


@app.command()
def availableModels():
    """
    Print a catalog of available pretrained models.

    This is a thin CLI wrapper around `scripts.trained_models.trained_models()`.
    It writes model information to stdout for human inspection.

    Usage:
        python main.py availablemodels
    """
    from scripts.trained_models import trained_models
    trained_models()

# ======================================
# available gpus as found by tensorflow
# ======================================

@app.command()
def available_gpus():
    """
    Print GPUs available to tranquillyzer

    Column descriptions:
        Raw Names   - Names as pulled from TensorFlow
        Clean Names - Cleaned names based on number of GPUs
    """
    print("Querying GPUs - this may take some time...")

    import scripts.available_gpus as available_gpus
    available_gpus.available_gpus()

# ===========================================
# extract reads, read_names from fasta file
# ===========================================


@app.command(no_args_is_help=True)
def preprocess(
    fasta_dir: str, 
    output_dir: str,
    output_base_qual: bool = typer.Option(False, help=(
        "Whether to output base quality scores"
        )),
    chunk_size: int = typer.Option(100000, help=(
        "Base chunk size, dynamically adjusts based on read length"
        )),
    threads: int = typer.Option(12, help=(
        "Number of CPU threads"
        ))
    ):
    """
    Preprocess raw FASTA/FASTQ files into length-binned Parquet files.

    Steps:
      1) Discover sequence files under `fasta_dir`.
      2) If one file: extract & bin reads serially into TSV chunks, then convert to Parquet.
         If multiple: run `parallel_preprocess_data` across `threads`.
      3) Record peak memory and runtime.

    Args:
        fasta_dir: Directory containing input FASTA/FASTQ (optionally gzipped) files.
        output_dir: Directory where outputs are written: `<output_dir>/full_length_pp_fa`.
        output_base_qual: If True, include base qualities in outputs when available.
        chunk_size: Base chunk size for binning; effective size scales with read-length distribution.
        threads: Number of CPU workers for parallel preprocessing.

    Outputs:
        - `<output_dir>/full_length_pp_fa/*.tsv[.lock]` (intermediate, cleaned up)
        - `<output_dir>/full_length_pp_fa/*.parquet` (final)
        - `<output_dir>/full_length_pp_fa/read_index.parquet` (read → parquet file mapping)
        - Logs to stdout/stderr.

    Raises:
        FileNotFoundError: If `fasta_dir` has no readable sequence files.
        RuntimeError: Propagated from subprocesses if failures occur.
    """
    from wrappers.preprocess_wrap import preprocess_wrap
    preprocess_wrap(fasta_dir, output_dir, output_base_qual, chunk_size, threads)

# ==============================
# plot read length distribution
# ==============================


@app.command(no_args_is_help=True)
def readlengthDist(output_dir: str):
    """
    Generate a read-length distribution plot from preprocessed parquet files.

    Reads the binned Parquet files in `<output_dir>/full_length_pp_fa` and
    writes a PNG/PDF plot into `<output_dir>/plots`.

    Args:
        output_dir: Base output directory from the `preprocess` step.

    Outputs:
        `<output_dir>/plots/read_length_distribution.*`

    Raises:
        FileNotFoundError: If required Parquet files are missing.
    """
    from wrappers.read_length_distr_wrap import read_length_distr_wrap
    read_length_distr_wrap(output_dir)

# ===========================================
# inspect selected reads for annotations
# ===========================================


@app.command(no_args_is_help=True)
def visualize(
    output_dir: str,          
    output_file: str = typer.Option(
        "full_read_annots",
        help=("""Output annotation file name.\n
        Extension .pdf will be added automatically""")),
        model_name: str = typer.Option(
            "10x3p_sc_ont_011",
            help="""Base model name. Use the name of the model without any suffix.\n
            For model-type CRF, _w_CRF will be added to the base model name"""),
        model_type: Annotated[
            str, typer.Option(help="""
            [red]REG[/red] = [green]CNN-LSTM[/green]\n
            [red]CRF[/red] = [green]CNN-LSTM-CRF[/green]
            """)
            ] = "CRF",
        seq_order_file: str = typer.Option(
            None,
            help="Path to the seq_orders.tsv file. If not provided, uses the default from utils."
            ),
        gpu_mem: Annotated[
            str, typer.Option(
                help="""
                Total memory of the GPU in GB.\n
                => If there's only one GPU or multiple-GPUs with same memory,
                specify an integer\n
                => If there are mutliple GPUs with different memories,
                specify a comma-separated list (e.g., 8,16,32)\n
                => If nothing is specified and one or more GPUs are available, 
                12 GB will be used by default.\n
                """
                )
            ] = None,
        target_tokens: Annotated[
            int, typer.Option(
                help="""Approximate token budget *per GPU replica* used to pick a safe batch size.\n
                => A 'token' is one input position after padding (for DNA here: ~1 base = 1 token).\n
                => Effective tokens per replica ≈ batch_size × padded_seq_len.\n
                => Increase to try larger batches (more memory), decrease if you hit OOM.\n
                => If running on CPU, this still guides batch size heuristics."""
                )
            ] = 1_200_000,
        vram_headroom: float = typer.Option(0.35, help="Fraction of GPU memory to reserve as headroom"),
        min_batch_size: int = typer.Option(1, help="Minimum batch size for model inference"),
        max_batch_size: int = typer.Option(2000, help="Maximum batch size for model inference"),
        num_reads: int = typer.Option(
            None,
            help="Number of reads to randomly visualize from each Parquet file."
            ),
        read_names: str = typer.Option(
            None,
            help="Comma-separated list of read names to visualize"),
        threads: int = typer.Option(2, help=(
            "Number of CPU threads"
            )
        )
    ):
    """
    Run model inference on selected reads and export per-read annotation plots.

    You can either provide `read_names` (comma-separated list) to visualize
    specific reads, or supply `num_reads` to randomly sample reads from the
    global `read_index.parquet`.

    Args:
        output_dir: Pipeline base output directory (expects `full_length_pp_fa/`).
        output_file: Base name for the output PDF under `<output_dir>/plots`.
        model_name: Base model name; `_w_CRF` suffix is inferred for CRF mode.
        model_type: One of {"REG","CRF"}; model choice for inference.
        gpu_mem: Optional GPU memory budget string (e.g., "12" or "8,16").
        target_tokens: Token budget per replica to estimate batch sizing.
        vram_headroom: Fraction of VRAM to reserve to reduce OOM risk.
        min_batch_size: Lower bound for batch size search.
        max_batch_size: Upper bound for batch size search.
        num_reads: If provided, randomly sample this many reads to visualize.
        read_names: If provided, comma-separated explicit read IDs to visualize.
        threads: CPU workers for downstream decoding/extraction.

    Outputs:
        `<output_dir>/plots/<output_file>.pdf` — tiled per-read annotation plots.

    Raises:
        ValueError: If neither `num_reads` nor `read_names` is supplied.
        FileNotFoundError: If required indices/files are missing.
    """
    from wrappers.visualize_wrap import visualize_wrap
    
    visualize_wrap(output_dir, output_file, model_name, model_type, 
                   seq_order_file, gpu_mem, target_tokens, vram_headroom, 
                   min_batch_size, max_batch_size, num_reads, read_names, threads)

# =========================
# Annotate all the reads
# =========================


@app.command(no_args_is_help=True)
def annotate_reads(
    output_dir: str,
    whitelist_file: str,
    output_fmt: str = typer.Option("fasta", help=(
        "output format for demultiplexed reads: fasta or fastq"
        )),
    model_name: str = typer.Option(
        "10x3p_sc_ont_011",
        help="""Base model name. Use the name of the model without any suffix.\n
        For model-type CRF, _w_CRF will be added to the base model name"""),
    model_type: Annotated[
        str, typer.Option(
            help="""
            [red]REG[/red] = [green]CNN-LSTM[/green]\n
            [red]CRF[/red] = [green]CNN-LSTM-CRF[/green]\n
            [red]HYB[/red] = [green]First pass with CNN-LSTM and second \n
            (of reads not qualifying validity filter) with CNN-LSTM-CRF[/green]
            """)
        ] = "HYB",
    seq_order_file: str = typer.Option(
        None,
        help="Path to the seq_orders.tsv file. If not provided, uses the default from utils."
    ),
    chunk_size: int = typer.Option(
        100000, help=(
        "Base chunk size, dynamically adjusts based on read length"
        )),
    gpu_mem: Annotated[
        str, typer.Option(
            help="""
            Total memory of the GPU in GB.\n
            => If there's only one GPU or multiple-GPUs with same memory, specify an integer\n
            => If there are mutliple GPUs with different memories, specify a comma-separated list (e.g., 8,16,32)\n
            => If nothing is specified and one or more GPUs are available, 12 GB will be used by default.\n
            """)
        ] = None,
    target_tokens: Annotated[
        int, typer.Option(help=
        """Approximate token budget *per GPU replica* used to pick a safe batch size.\n
        => A 'token' is one input position after padding (for DNA here: ~1 base = 1 token).\n
        => Effective tokens per replica ≈ batch_size × padded_seq_len.\n
        => Increase to try larger batches (more memory), decrease if you hit OOM.\n
        => If running on CPU, this still guides batch size heuristics."""
        )] = 1_200_000,
    vram_headroom: float = typer.Option(0.35, help="Fraction of GPU memory to reserve as headroom"),
    min_batch_size: int = typer.Option(1, help="Minimum batch size for model inference"),
    max_batch_size: int = typer.Option(8192, help="Maximum batch size for model inference"),
    bc_lv_threshold: int = typer.Option(2, help="lv-distance threshold for barcode correction"),
    threads: int = typer.Option(12, help="Number of CPU threads for barcode correction and demultiplexing"),
    max_queue_size: int = typer.Option(3, help="Max number of Parquet files to queue for post-processing"),
):
    """
    End-to-end annotation, barcode correction, demultiplexing, and QC plots.

    Pipeline:
      1) Iterate through binned Parquet files and run model predictions.
      2) Post-process predictions to call segments, correct barcodes (Levenshtein),
         and write demultiplexed reads to FASTA/FASTQ.
      3) Optionally (HYB) re-run invalid reads with CRF model.
      4) Emit summary TSV/Parquet files and PDF plots for barcode & demux stats.

    Args:
        output_dir: Base directory with `full_length_pp_fa/` and target for outputs.
        whitelist_file: TSV with valid barcode columns; used for demultiplexing.
        output_fmt: "fasta" or "fastq" for demultiplexed outputs.
        model_name: Base model label (without `_w_CRF`).
        model_type: "REG", "CRF", or "HYB" (REG pass then CRF on invalid).
        chunk_size: Row group size for final Parquet conversions.
        gpu_mem: Optional GPU memory budget string (e.g., "12" or "8,16").
        target_tokens: Token budget per replica to guide batching.
        vram_headroom: Fraction of VRAM to keep free as safety margin.
        min_batch_size: Min batch size for inference.
        max_batch_size: Max batch size for inference.
        bc_lv_threshold: Levenshtein threshold for barcode correction.
        threads: CPU workers for barcode/demux post-processing.
        max_queue_size: Max in-flight Parquet chunks for worker queueing.

    Outputs:
        - `<output_dir>/annotations_valid.parquet` and `_invalid.parquet`
        - `<output_dir>/demuxed_fasta/demuxed.(fa|fq)` and `ambiguous.(fa|fq)`
        - `<output_dir>/plots/barcode_plots.pdf`, `demux_plots.pdf`
        - Read length & cDNA length plots for valid reads
        - Match/cell counts TSVs

    Raises:
        FileNotFoundError: If expected input files or whitelist are missing.
        TimeoutError: If worker result collection stalls beyond threshold.
        RuntimeError: Propagated exceptions from worker processes.
    """
    from wrappers.annotate_reads_wrap import annotate_reads_wrap
    annotate_reads_wrap(output_dir, whitelist_file, output_fmt, 
                        model_name, model_type, seq_order_file,
                        chunk_size, gpu_mem, target_tokens,
                        vram_headroom, min_batch_size, max_batch_size,
                        bc_lv_threshold, threads, max_queue_size)

# ======================================
# align inserts to the reference genome
# ======================================


@app.command(no_args_is_help=True)
def align(
    input_dir: str,
    ref: str,
    output_dir: str,
    preset: str = typer.Option("splice", help="minimap2 preset"),
    filt_flag: str = typer.Option("0", help=(
        "Flag for filtering out (-F in samtools) the reads. "
        "Default is 260, to filter out secondary alignments "
        "and unmapped reads."
        )),
    mapq: int = typer.Option(0, help=(
        "minimap mapq for the alignments to be included "
        "for the downstream analysis"
        )),
    threads: int = typer.Option(12, help="number of CPU threads"),
    add_minimap_args: str = typer.Option("", help=(
        "additional minimap2 arguments"
        ))
):
    """
    Align demultiplexed reads to a reference with minimap2 and index the BAM.

    This command finds `<input_dir>/demuxed_fasta/demuxed.(fastq|fasta)`,
    aligns using `minimap2 -ax <preset>`, filters with samtools (`-F filt_flag -q mapq`),
    sorts and writes a coordinate-sorted BAM, then creates a BAM index.

    Args:
        input_dir: Directory containing `demuxed_fasta` outputs.s
        ref: Reference genome/transcriptome FASTA for minimap2.
        output_dir: Base directory to write `aligned_files/demuxed_aligned.bam`.
        preset: Minimap2 preset (e.g., "splice" for long RNA).
        filt_flag: Samtools view `-F` flag to drop reads by bitwise flags.
        mapq: Minimum MAPQ to keep alignments.
        threads: Number of CPU threads for minimap2/samtools.
        add_minimap_args: Extra args appended to the minimap2 command.

    Outputs:
        `<output_dir>/aligned_files/demuxed_aligned.bam[.bai]`

    Raises:
        FileNotFoundError: If demultiplexed FASTA/FASTQ cannot be located.
        CalledProcessError: If minimap2/samtools commands fail.
    """
    from wrappers.align_wrap import align_wrap
    align_wrap(input_dir, ref, output_dir, preset, 
               filt_flag, mapq, threads, add_minimap_args)

# ==============================
# Deduplication using UMI-tools
# ==============================


@app.command(no_args_is_help=True)
def dedup(
    input_dir: str,
    lv_threshold: int = typer.Option(2, help=(
        "levenshtein distance threshold for UMI similarity"
        )),
    stranded: bool = typer.Option(True, help=(
        "if directional or non-directional library"
        )),
    per_cell: bool = typer.Option(True, help=(
        "whether to correct umi's per cell basis"
        )),
    threads: int = typer.Option(12, help="number of CPU threads")
):
    """
    Mark/remove PCR duplicates using UMI-aware clustering on aligned reads.

    Invokes the project-specific `deduplication_parallel` which clusters UMIs
    by Levenshtein distance (threshold `lv_threshold`), optionally per-cell and
    respecting library strandedness.

    Args:
        input_dir: Base directory with `aligned_files/demuxed_aligned.bam`.
        lv_threshold: Edit distance threshold for grouping similar UMIs.
        stranded: Whether library is strand-aware.
        per_cell: If True, perform UMI correction per cell ID.
        threads: CPU threads used by the deduplication pipeline.

    Outputs:
        `<input_dir>/aligned_files/demuxed_aligned_dup_marked.bam[.bai]`

    Raises:
        FileNotFoundError: If expected output BAM is not produced.
        RuntimeError: If underlying tools throw errors.
    """
    from wrappers.dedup_wrap import dedup_wrap
    dedup_wrap(input_dir, lv_threshold, stranded, per_cell, threads)

# ===========================
# Simulate training dataset
# ===========================


@app.command(no_args_is_help=True)
def simulate_data(model_name: str,
                  output_dir: str,
                  training_seq_orders_file: str = typer.Option(None, help=(
                      "Path to the seq_orders.tsv file. If not provided, "
                      "uses the default from utils."
                  )),
                  num_reads: int = typer.Option(50000, help="number of reads to simulate"),
                  mismatch_rate: float = typer.Option(0.05, help="mismatch rate"),
                  insertion_rate: float = typer.Option(0.05, help="insertion rate"),
                  deletion_rate: float = typer.Option(0.06, help="deletion rate"),
                  min_cDNA: int = typer.Option(100, help="minimum cDNA length"),
                  max_cDNA: int = typer.Option(500, help="maximum cDNA length"),
                  polyT_error_rate: float = typer.Option(0.02, help=(
                      "error rate within "
                      "polyT or polyA segments"
                      )),
                  max_insertions: float = typer.Option(1, help=(
                      "maximum number of allowed "
                      "insertions after a base"
                      )),
                  threads: int = typer.Option(2, help="number of CPU threads"),
                  rc: bool = typer.Option(True, help=(
                      "whether to include reverse complements of the reads in "
                      "the training data.\nFinal dataset "
                      "will contain twice the number of user-specified reads"
                      )),
                  transcriptome: str = typer.Option(None, help="transcriptome fasta file"),
                  invalid_fraction: float = typer.Option(0.3, help="fraction of invalid reads to generate")):
    """
    Generate synthetic labeled reads for training, and serialize to PKL.

    The generator uses a model-specific segment order/pattern and either
    provided transcripts or synthetic random transcripts to create realistic
    reads with configurable error profiles.

    Args:
        model_name: Model key for segment order specification.
        output_dir: Destination for `simulated_data/reads.pkl` and `labels.pkl`.
        num_reads: Number of primary reads to synthesize (before RC doubling).
        mismatch_rate: Base substitution probability.
        insertion_rate: Base insertion probability.
        deletion_rate: Base deletion probability.
        min_cDNA: Minimum transcript (cDNA) segment length.
        max_cDNA: Maximum transcript (cDNA) segment length.
        polyT_error_rate: Error rate within polyT/polyA regions.
        max_insertions: Maximum insertions per base position.
        threads: CPU threads used within the generator.
        rc: If True, include reverse complements (doubling dataset size).
        transcriptome: Optional FASTA of transcripts; otherwise random.
        invalid_fraction: Fraction of reads to synthesize as invalid.

    Outputs:
        `<output_dir>/simulated_data/reads.pkl`
        `<output_dir>/simulated_data/labels.pkl`
    """
    from wrappers.simulate_data_wrap import simulate_data_wrap
    simulate_data_wrap(model_name, output_dir, training_seq_orders_file,
                       num_reads, mismatch_rate, insertion_rate, deletion_rate,
                       min_cDNA, max_cDNA, polyT_error_rate, max_insertions,
                       threads, rc, transcriptome, invalid_fraction)

# ===============
#  Train model
# ===============


@app.command(no_args_is_help=True)
def train_model(model_name: str,
                output_dir: str,
                param_file: str = typer.Option(None, help=(
                    "Path to the training_params.tsv file. If not provided, "
                    "uses the default from utils."
                )),
                training_seq_orders_file: str = typer.Option(None, help=(
                    "Path to the seq_orders.tsv file. If not provided, "
                    "uses the default from utils."
                )),
                num_val_reads: int = typer.Option(20, help="number of reads to simulate"),
                mismatch_rate: float = typer.Option(0.05, help="mismatch rate"),
                insertion_rate: float = typer.Option(0.05, help="insertion rate"),
                deletion_rate: float = typer.Option(0.06, help="deletion rate"),
                min_cDNA: int = typer.Option(100, help="minimum cDNA length"),
                max_cDNA: int = typer.Option(500, help="maximum cDNA length"),
                polyT_error_rate: float = typer.Option(0.02, help="error rate within polyT or polyA segments"),
                max_insertions: float = typer.Option(2, help="maximum number of allowed insertions after a base"),
                threads: int = typer.Option(2, help="number of CPU threads"),
                rc: bool = typer.Option(True, help=(
                    "whether to include reverse complements of "
                    "the reads in the training data.\nFinal dataset will "
                    "contain twice the number of user-specified reads"
                    )),
                transcriptome: str = typer.Option(None, help="transcriptome fasta file"),
                invalid_fraction: float = typer.Option(0.3, help="fraction of invalid reads to generate"),
                gpu_mem: Annotated[
                str, typer.Option(
                    help="""
                    Total memory of the GPU in GB.\n
                    => If there's only one GPU or multiple-GPUs with same memory, 
                    specify an integer\n
                    => If there are mutliple GPUs with different memories,
                    specify a comma-separated list (e.g., 8,16,32)\n
                    => If nothing is specified and one or more GPUs are available, 
                    12 GB will be used by default.\n
                    """)] = None,
                target_tokens: Annotated[
                    int, typer.Option(help=
                    """Approximate token budget *per GPU replica* used to pick a safe batch size.\n
                    => A 'token' is one input position after padding (for DNA here: ~1 base = 1 token).\n
                    => Effective tokens per replica ≈ batch_size × padded_seq_len.\n
                    => Increase to try larger batches (more memory), decrease if you hit OOM.\n
                    => If running on CPU, this still guides batch size heuristics."""
                    )] = 1_200_000,
                vram_headroom: float = typer.Option(0.35, help="Fraction of GPU memory to reserve as headroom"),
                min_batch_size: int = typer.Option(1, help="Minimum batch size for model inference"),
                max_batch_size: int = typer.Option(2000, help="Maximum batch size for model inference"),):
    """
    Grid-train model variants from parameter table and export artifacts.

    For a given `model_name`, this reads `utils/training_params.tsv`,
    enumerates parameter combinations, trains each variant using a distributed
    strategy when available, and saves:
      - model weights / SavedModel
      - fitted label binarizer
      - training history
      - validation visualization PDF on a small synthetic set

    Args:
        model_name: Column in `training_params.tsv` whose parameter grid to use.
        output_dir: Base directory to write per-variant subfolders and artifacts.
        num_val_reads: Number of validation reads to synthesize.
        mismatch_rate / insertion_rate / deletion_rate: Error model for validation set.
        min_cDNA / max_cDNA: Transcript length bounds for validation set.
        polyT_error_rate: Error rate inside polyT/polyA for validation set.
        max_insertions: Max insertions per position for validation set.
        threads: CPU threads for validation read synthesis.
        rc: Include reverse complements for validation set.
        transcriptome: Optional validation FASTA; else random transcripts.
        invalid_fraction: Fraction of invalid reads to generate for validation set.
        gpu_mem / target_tokens / vram_headroom / min_batch_size / max_batch_size:
            Parameters to guide validation inference batch sizing (plot stage).

    Outputs (per variant `<model_name>_<idx>`):
        - `*.h5` or SavedModel
        - `<model_name>_<idx>_lbl_bin.pkl`
        - `<model_name>_<idx>_history.tsv`
        - `<model_name>_<idx>_val_viz.pdf`

    Raises:
        FileNotFoundError: If `training_params.tsv` missing or `model_name` not present.
        RuntimeError: If training fails.
    """
    from wrappers.train_model_wrap import train_model_wrap
    train_model_wrap(model_name, output_dir, param_file, training_seq_orders_file,
                     num_val_reads, mismatch_rate, insertion_rate, deletion_rate,
                     min_cDNA, max_cDNA, polyT_error_rate, max_insertions,
                     threads, rc, transcriptome, invalid_fraction, gpu_mem,
                     target_tokens, vram_headroom, min_batch_size, max_batch_size)


if __name__ == "__main__":
    app()
