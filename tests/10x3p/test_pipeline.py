import os
import subprocess
from pathlib import Path
import pytest

RAW_INPUT_DIR = Path("tests/10x3p/data")
OUT_DIR = Path("tests/10x3p")
SIM_DIR = Path("tests/10x3p")
BARCODES = Path("tests/10x3p/barcodes.tsv")
REF_FASTA = Path("tests/references/hg38_gencode_chr21.fa")
transcriptome = Path("tests/references/")
THREADS = 1

OUT_DIR.mkdir(exist_ok=True, parents=True)

COVRC = str(Path(".coveragerc").resolve())
COVDATA = str(Path(".coverage").resolve())


def run_cmd(cmd, timeout=900):
    env = os.environ.copy()
    env["COVERAGE_PROCESS_START"] = COVRC
    env["COVERAGE_FILE"] = COVDATA
    env.setdefault("PYTHONPATH", os.getcwd())

    print(f"\n>> Running: {' '.join(map(str, cmd))}")
    p = subprocess.run(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    print(p.stdout)
    print(p.stderr)
    assert p.returncode == 0, f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}"


@pytest.mark.order(1)
def test_preprocess_wo_base_qual():
    run_cmd(
        [
            "tranquillyzer",
            "preprocess",
            RAW_INPUT_DIR,
            OUT_DIR,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(2)
def test_readlengthdist():
    run_cmd(
        [
            "tranquillyzer",
            "readlengthdist",
            OUT_DIR,
        ]
    )


@pytest.mark.order(3)
def test_visualize():
    run_cmd(
        [
            "tranquillyzer",
            "visualize",
            OUT_DIR,
            "--output-file",
            "test_visualization",
            "--model-type",
            "CRF",
            "--num-reads",
            "10",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(4)
def test_annotate_reads():
    run_cmd(
        [
            "tranquillyzer",
            "annotate-reads",
            OUT_DIR,
            BARCODES,
            "--model-type",
            "CRF",
            "--chunk-size",
            100000,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(5)
def test_align():
    run_cmd(
        [
            "tranquillyzer",
            "align",
            OUT_DIR,
            REF_FASTA,
            OUT_DIR,
            "--preset",
            "splice",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(6)
def test_dedup():
    run_cmd(
        [
            "tranquillyzer",
            "dedup",
            OUT_DIR,
            "--lv-threshold",
            1,
            "--threads",
            THREADS,
            "--per-cell",
        ]
    )


@pytest.mark.order(7)
def test_split_bam():
    run_cmd(
        [
            "tranquillyzer",
            "split-bam",
            f"{OUT_DIR}/aligned_files/demuxed_aligned_dup_marked.bam",
            "--bucket-threads",
            1,
            "--merge-threads",
            1,
            "--max-open-cb-writers",
            500,
            "--filter-secondary",
            "--filter-supplementary",
        ]
    )


@pytest.mark.order(8)
def test_simulate_data():
    run_cmd(
        [
            "tranquillyzer",
            "simulate-data",
            "10x3p_sc_ont",
            SIM_DIR,
            "--num-reads",
            1000,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(9)
def test_available_models():
    run_cmd(
        [
            "tranquillyzer",
            "availablemodels",
        ]
    )


@pytest.mark.order(10)
def test_preprocess_w_base_qual():
    run_cmd(
        [
            "tranquillyzer",
            "preprocess",
            RAW_INPUT_DIR,
            OUT_DIR,
            "--output-base-qual",
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(11)
def test_annotate_reads_w_base_qual():
    run_cmd(
        [
            "tranquillyzer",
            "annotate-reads",
            OUT_DIR,
            BARCODES,
            "--output-fmt",
            "fastq",
            "--model-type",
            "CRF",
            "--chunk-size",
            100000,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(12)
def test_train_model():
    run_cmd(
        [
            "tranquillyzer",
            "train-model",
            "10x3p_sc_ont",
            SIM_DIR,
            "--threads",
            THREADS,
        ]
    )


@pytest.mark.order(13)
def test_available_gpus():
    run_cmd(
        [
            "tranquillyzer",
            "available-gpus",
        ]
    )
