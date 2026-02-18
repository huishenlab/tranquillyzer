import os
import shutil
import time
import logging
import pysam
import tempfile
import zlib

from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Utilities
# ----------------------------


def _stable_bucket_id(cb: str, nbuckets: int) -> int:
    """Deterministic bucket id for a CB tag (stable across processes)."""
    return (zlib.crc32(cb.encode("utf-8")) & 0xFFFFFFFF) % nbuckets


def _safe_cb_filename(cb: str) -> str:
    """Make CB safe for filenames."""
    return cb.replace(":", "_").replace("/", "_")


def _validate_split_params(
    input_bam: str,
    out_dir: str,
    nbuckets: int,
    tag: str,
    max_open_cb_writers: int,
    bucket_threads: Optional[int],
    merge_threads: Optional[int],
) -> None:
    if not os.path.exists(input_bam):
        raise FileNotFoundError(f"Input BAM not found: {input_bam}")
    if nbuckets <= 0:
        raise ValueError("nbuckets must be > 0")
    if (nbuckets & (nbuckets - 1)) != 0:
        # not required, but nice to call out since powers of two bucket well
        logger.warning(
            "nbuckets=%d is not a power of two; that's fine but 128/256/512 often bucket nicely.",
            nbuckets,
        )
    if not tag or not isinstance(tag, str):
        raise ValueError("tag must be a non-empty string")
    if max_open_cb_writers <= 0:
        raise ValueError("max_open_cb_writers must be > 0")
    if bucket_threads is not None and bucket_threads <= 0:
        raise ValueError("bucket_threads must be > 0 or None")
    if merge_threads is not None and merge_threads <= 0:
        raise ValueError("merge_threads must be > 0 or None")

    os.makedirs(out_dir, exist_ok=True)


def ensure_sorted_and_indexed(bam_path: str, prefer_csi: bool = False) -> str:
    """
    Ensure BAM is coordinate-sorted and indexed.
    Returns path to sorted+indexed BAM (may be a new file).
    """
    t0 = time.time()
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        so = (bam.header or {}).get("HD", {}).get("SO", "")
        is_coord_sorted = so == "coordinate"

    out_bam = bam_path
    if not is_coord_sorted:
        base, ext = os.path.splitext(bam_path)
        out_bam = base + ".sorted" + ext
        logger.info(
            "Input BAM not coordinate-sorted (SO=%r). Sorting -> %s",
            so,
            out_bam,
        )
        pysam.sort("-o", out_bam, bam_path)
    else:
        logger.info("Input BAM is coordinate-sorted (SO=coordinate).")

    bai = out_bam + ".bai"
    csi = os.path.splitext(out_bam)[0] + ".csi"
    if not (os.path.exists(bai) or os.path.exists(csi)):
        logger.info(
            "Index not found for %s. Creating %s index...",
            out_bam,
            "CSI" if prefer_csi else "BAI",
        )
        if prefer_csi:
            pysam.index(out_bam, min_shift=14)
        else:
            pysam.index(out_bam)
    else:
        logger.info("Index found for %s.", out_bam)

    logger.info("ensure_sorted_and_indexed finished in %.1fs", time.time() - t0)
    return out_bam


# ----------------------------
# LRU cache for output CB writers
# ----------------------------


class WriterCache:
    """
    LRU cache for pysam.AlignmentFile writers.
    Prevents "too many open files" when writing many output BAMs.

    Important:
      Writers are opened in "ab" (append-binary) mode if the
      file already exists. This is critical because bucket
      processing may write to the same CB file across time
      (different buckets are disjoint by CB, but re-entry can
      happen if you rerun or keep tmp around).
    """

    def __init__(self, max_open: int, header_dict: dict, out_dir: str):
        self.max_open = max_open
        self.header_dict = header_dict
        self.out_dir = out_dir
        self._writers: OrderedDict[str, pysam.AlignmentFile] = OrderedDict()
        self.opened_total = 0
        self.evicted_total = 0

    def _open_writer(self, cb: str) -> pysam.AlignmentFile:
        safe = _safe_cb_filename(cb)
        out_bam = os.path.join(self.out_dir, f"{safe}.bam")
        mode = "ab" if os.path.exists(out_bam) else "wb"
        w = pysam.AlignmentFile(out_bam, mode, header=self.header_dict)
        self.opened_total += 1
        return w

    def get(self, cb: str) -> pysam.AlignmentFile:
        if cb in self._writers:
            w = self._writers.pop(cb)
            self._writers[cb] = w  # mark most-recent
            return w

        # evict if needed
        if len(self._writers) >= self.max_open:
            _, oldw = self._writers.popitem(last=False)
            oldw.close()
            self.evicted_total += 1

        w = self._open_writer(cb)
        self._writers[cb] = w
        return w

    def close_all(self):
        for w in self._writers.values():
            w.close()
        self._writers.clear()


# ----------------------------
# Stage 1: Per-contig workers
# write bucket BAM parts
# ----------------------------


def _process_ref_to_buckets(args) -> Tuple[str, List[int], Dict[str, int]]:
    """
    Worker: process one reference/contig.
    Writes reads into nbuckets bucket BAMs for that contig.
    Output files: tmp_dir / f"bucket{bid:04d}.{ref}.bam"

    Returns:
      (ref, sorted(list_of_seen_bucket_ids), stats_dict)
    """
    (
        ref,
        bam_path,
        tmp_dir,
        nbuckets,
        tag,
        filter_secondary,
        filter_supplementary,
        filter_unmapped,
        filter_duplicates,
        min_mapq,
    ) = args

    os.makedirs(tmp_dir, exist_ok=True)

    stats = {
        "ref": ref,
        "n_in": 0,
        "n_kept": 0,
        "n_no_tag": 0,
        "n_filtered": 0,
        "n_open_buckets": 0,
    }

    t0 = time.time()

    with pysam.AlignmentFile(bam_path, "rb") as bam_in:
        header_dict = bam_in.header.to_dict()

        bucket_writers: Dict[int, pysam.AlignmentFile] = {}
        seen_buckets = set()

        def get_bucket_writer(bid: int) -> pysam.AlignmentFile:
            if bid in bucket_writers:
                return bucket_writers[bid]
            path = os.path.join(tmp_dir, f"bucket{bid:04d}.{ref}.bam")
            w = pysam.AlignmentFile(path, "wb", header=header_dict)
            bucket_writers[bid] = w
            return w

        try:
            for read in bam_in.fetch(ref):
                stats["n_in"] += 1

                # Filtering
                if filter_unmapped and read.is_unmapped:
                    stats["n_filtered"] += 1
                    continue
                if filter_secondary and read.is_secondary:
                    stats["n_filtered"] += 1
                    continue
                if filter_supplementary and read.is_supplementary:
                    stats["n_filtered"] += 1
                    continue
                if filter_duplicates and read.is_duplicate:
                    stats["n_filtered"] += 1
                    continue
                if (min_mapq is not None) and (read.mapping_quality < min_mapq):
                    stats["n_filtered"] += 1
                    continue

                if not read.has_tag(tag):
                    stats["n_no_tag"] += 1
                    continue

                cb = read.get_tag(tag)
                bid = _stable_bucket_id(cb, nbuckets)

                w = get_bucket_writer(bid)
                w.write(read)
                seen_buckets.add(bid)
                stats["n_kept"] += 1
        finally:
            for w in bucket_writers.values():
                w.close()

    stats["n_open_buckets"] = len(seen_buckets)

    # Worker logs (kept concise; parent also logs global progress)
    logger.info(
        "[stage1] %s: in=%d kept=%d no_%s=%d filtered=%d buckets=%d (%.1fs)",
        ref,
        stats["n_in"],
        stats["n_kept"],
        tag,
        stats["n_no_tag"],
        stats["n_filtered"],
        stats["n_open_buckets"],
        time.time() - t0,
    )

    return ref, sorted(seen_buckets), stats


# ----------------------------
# Stage 2: For each bucket, stream contig-parts in ref order
# and write final per-CB BAMs (LRU writer cache)
# ----------------------------


def _build_cb_bams_from_bucket(
    bucket_id: int,
    ref_order: List[str],
    tmp_dir: str,
    out_dir: str,
    header_dict: dict,
    tag: str,
    max_open_cb_writers: int,
    index_outputs: bool,
) -> Dict[str, int]:
    """
    Process a single bucket:
      - stream bucket{bucket_id}.{ref}.bam for each ref in ref_order
      - for each record, write to its CB output BAM
    Since the stream is coordinate-sorted, each CB's output BAM
    remains coordinate-sorted.

    Returns stats dict.
    """
    t0 = time.time()
    cache = WriterCache(max_open=max_open_cb_writers, header_dict=header_dict, out_dir=out_dir)

    stats = {
        "bucket_id": bucket_id,
        "parts_seen": 0,
        "records_in": 0,
        "records_written": 0,
        "cbs_seen": 0,
        "index_ok": 0,
        "index_fail": 0,
        "writers_opened_total": 0,
        "writers_evicted_total": 0,
    }

    seen_cbs = set()

    try:
        for ref in ref_order:
            part = os.path.join(tmp_dir, f"bucket{bucket_id:04d}.{ref}.bam")
            if not (os.path.exists(part) and os.path.getsize(part) > 0):
                continue

            stats["parts_seen"] += 1
            with pysam.AlignmentFile(part, "rb") as pin:
                for rec in pin.fetch(until_eof=True):
                    stats["records_in"] += 1
                    if not rec.has_tag(tag):
                        continue
                    cb = rec.get_tag(tag)
                    w = cache.get(cb)
                    w.write(rec)
                    seen_cbs.add(cb)
                    stats["records_written"] += 1
    finally:
        cache.close_all()

    stats["cbs_seen"] = len(seen_cbs)
    stats["writers_opened_total"] = cache.opened_total
    stats["writers_evicted_total"] = cache.evicted_total

    if index_outputs and seen_cbs:
        for cb in seen_cbs:
            safe = _safe_cb_filename(cb)
            out_bam = os.path.join(out_dir, f"{safe}.bam")
            try:
                pysam.index(out_bam)
                stats["index_ok"] += 1
            except Exception as e:
                stats["index_fail"] += 1
                logger.warning("[stage2] indexing failed: %s (%s)", out_bam, e)

    logger.info(
        "[stage2] bucket=%04d parts=%d rec_in=%d rec_out=%d cbs=%d opened=%d evicted=%d idx_ok=%d idx_fail=%d (%.1fs)",
        bucket_id,
        stats["parts_seen"],
        stats["records_in"],
        stats["records_written"],
        stats["cbs_seen"],
        stats["writers_opened_total"],
        stats["writers_evicted_total"],
        stats["index_ok"],
        stats["index_fail"],
        time.time() - t0,
    )

    return stats


def split_bam_file(
    input_bam: str,
    out_dir: str,
    bucket_threads: Optional[int] = None,
    merge_threads: Optional[int] = None,
    nbuckets: int = 256,
    tag: str = "CB",
    max_open_cb_writers: int = 128,
    filter_secondary: bool = False,
    filter_supplementary: bool = False,
    filter_unmapped: bool = True,
    filter_duplicates: bool = True,
    min_mapq: Optional[int] = None,
    keep_tmp: bool = False,
    index_outputs: bool = True,
    prefer_csi_index: bool = False,
):
    """
    Scalable split of a coordinate-sorted BAM into one BAM per CB tag.

    - Stage 1 (parallel by contig): write into a FIXED number of
    bucket BAMs per contig.
        => bounded open files + bounded temp file count.
    - Stage 2 (parallel by bucket): stream bucket parts in contig order and
    write final per-CB BAMs
        using an LRU cache so we never keep too many CB output files open.

    Output per-CB BAMs remain coordinate-sorted because we write reads in
    coordinate-sorted stream order.
    """
    _validate_split_params(
        input_bam=input_bam,
        out_dir=out_dir,
        nbuckets=nbuckets,
        tag=tag,
        max_open_cb_writers=max_open_cb_writers,
        bucket_threads=bucket_threads,
        merge_threads=merge_threads,
    )

    t_all = time.time()
    logger.info("Starting split_bam")
    logger.info("Input: %s", input_bam)
    logger.info("Out dir: %s", out_dir)
    logger.info(
        "Params: nbuckets=%d tag=%s bucket_threads=%s merge_threads=%s max_open_cb_writers=%d index_outputs=%s",
        nbuckets,
        tag,
        str(bucket_threads),
        str(merge_threads),
        max_open_cb_writers,
        index_outputs,
    )

    input_bam = ensure_sorted_and_indexed(input_bam, prefer_csi=prefer_csi_index)

    # Unique tmp dir so parallel runs don't collide
    tmp_dir = tempfile.mkdtemp(prefix="cb_split_tmp_", dir=out_dir)
    logger.info("Temp dir: %s", tmp_dir)

    with pysam.AlignmentFile(input_bam, "rb") as bam:
        ref_order = list(bam.references)
        header_dict = bam.header.to_dict()

    if not ref_order:
        raise ValueError("No references/contigs found in BAM header.")

    # -------- Stage 1 --------
    t1 = time.time()
    procs1 = min(bucket_threads or cpu_count(), len(ref_order)) or 1
    logger.info("[stage1] contigs=%d procs=%d", len(ref_order), procs1)

    args1 = [
        (
            ref,
            input_bam,
            tmp_dir,
            nbuckets,
            tag,
            filter_secondary,
            filter_supplementary,
            filter_unmapped,
            filter_duplicates,
            min_mapq,
        )
        for ref in ref_order
    ]

    if procs1 > 1:
        with Pool(procs1) as pool:
            stage1_results = pool.map(_process_ref_to_buckets, args1)
    else:
        stage1_results = [_process_ref_to_buckets(args) for args in args1]

    used_buckets = set()
    stage1_stats = []
    for _, bids, stats in stage1_results:
        used_buckets.update(bids)
        stage1_stats.append(stats)
    used_buckets = sorted(used_buckets)

    n_in_total = sum(s["n_in"] for s in stage1_stats)
    n_kept_total = sum(s["n_kept"] for s in stage1_stats)
    n_notag_total = sum(s["n_no_tag"] for s in stage1_stats)
    n_filt_total = sum(s["n_filtered"] for s in stage1_stats)

    logger.info(
        "[stage1] done in %.1fs | total_in=%d kept=%d no_%s=%d filtered=%d | buckets_used=%d/%d",
        time.time() - t1,
        n_in_total,
        n_kept_total,
        tag,
        n_notag_total,
        n_filt_total,
        len(used_buckets),
        nbuckets,
    )

    if not used_buckets:
        logger.warning("No buckets contained any reads with tag %s. Nothing to write.", tag)
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # -------- Stage 2 --------
    t2 = time.time()
    # I/O heavy: default to <=8 unless user overrides
    procs2_default = max(1, min(cpu_count(), 8))
    procs2 = min(merge_threads or procs2_default, len(used_buckets)) or 1
    logger.info(
        "[stage2] buckets=%d procs=%d max_open_cb_writers=%d",
        len(used_buckets),
        procs2,
        max_open_cb_writers,
    )

    stage2_args = [
        (
            bid,
            ref_order,
            tmp_dir,
            out_dir,
            header_dict,
            tag,
            max_open_cb_writers,
            index_outputs,
        )
        for bid in used_buckets
    ]

    if procs2 > 1:
        with Pool(procs2) as pool:
            stage2_stats_list = pool.starmap(_build_cb_bams_from_bucket, stage2_args)
    else:
        stage2_stats_list = [_build_cb_bams_from_bucket(*arg) for arg in stage2_args]

    # Summarize stage2
    total_bucket_parts = sum(s["parts_seen"] for s in stage2_stats_list)
    total_records_in = sum(s["records_in"] for s in stage2_stats_list)
    total_records_written = sum(s["records_written"] for s in stage2_stats_list)
    total_cbs_seen = sum(s["cbs_seen"] for s in stage2_stats_list)
    idx_ok = sum(s["index_ok"] for s in stage2_stats_list)
    idx_fail = sum(s["index_fail"] for s in stage2_stats_list)

    logger.info(
        "[stage2] done in %.1fs | parts=%d rec_in=%d rec_out=%d cbs_total~=%d idx_ok=%d idx_fail=%d",
        time.time() - t2,
        total_bucket_parts,
        total_records_in,
        total_records_written,
        total_cbs_seen,
        idx_ok,
        idx_fail,
    )

    # Cleanup
    if keep_tmp:
        logger.info("Keeping temp dir: %s", tmp_dir)
    else:
        logger.info("Removing temp dir: %s", tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Finished split_bam in %.1fs", time.time() - t_all)
