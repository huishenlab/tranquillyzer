import logging

logger = logging.getLogger(__name__)


def dedup_wrap(input_dir, lv_threshold, stranded, per_cell, threads):
    import os
    import time
    import resource
    import pysam
    import subprocess

    from scripts.deduplicate import deduplication_parallel

    start = time.time()
    logger.info("Starting duplicate marking process")

    aligned_bam = "aligned_files/demuxed_aligned.bam"
    dup_marked_bam = "aligned_files/demuxed_aligned_dup_marked.bam"

    input_bam = os.path.join(input_dir, aligned_bam)
    out_bam = os.path.join(input_dir, dup_marked_bam)

    deduplication_parallel(input_bam, out_bam, lv_threshold, per_cell, threads, stranded)

    if not os.path.exists(out_bam):
        raise FileNotFoundError(f"Expected output BAM not found: {out_bam}")

    # Check BAM sort order and sort if needed before indexing
    so = None
    try:
        with pysam.AlignmentFile(out_bam, "rb") as bam:
            so = (bam.header.get("HD") or {}).get("SO")
    except Exception as e:
        logger.warning(f"Could not read BAM header to check sort order ({e}). Assuming unsorted.")

    # Use `so` so it's not "unused" + ensure correct indexing behavior
    needs_sort = (so is None) or (str(so).lower() != "coordinate")
    if needs_sort:
        logger.info(f"BAM sort order is {so!r}; sorting to coordinate order before indexing.")
        sorted_bam = out_bam.replace(".bam", ".sorted.bam")
        subprocess.run(
            ["samtools", "sort", "-@", str(threads), "-o", sorted_bam, out_bam],
            check=True,
        )
        os.replace(sorted_bam, out_bam)
        # Optional: ensure header says coordinate (samtools sort typically sets it)
        so = "coordinate"

    logger.info(f"Indexing duplicate marked BAM file (SO={so!r})")
    subprocess.run(["samtools", "index", "-@", str(threads), out_bam], check=True)
    logger.info(f"Indexing completed for {out_bam}")

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss  # Linux gives KB
    logger.info(f"Peak memory usage: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
