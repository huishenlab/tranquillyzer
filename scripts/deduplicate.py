import os
import logging
import multiprocessing as mp
from collections import defaultdict, deque
from dataclasses import dataclass

import pysam
from rapidfuzz.distance import Hamming, Levenshtein

logger = logging.getLogger(__name__)


def extract_cb_umi(read_name: str):
    parts = read_name.split("_")
    if len(parts) >= 3:
        # CBC and UMI are at the end
        cb = parts[-2]
        umi = parts[-1]

        # Remove CBC and UMI from the read name
        cleaned_name = "_".join(parts[:-2])

        return cb, umi, cleaned_name

    return None, None, read_name


# UMI distance: fast and indel-safe
def _umi_dist(a: str, b: str, max_d: int) -> int:
    """
    Compute edit distance between two UMIs with early cutoff.
    - Uses fast Hamming when lengths match.
    - Falls back to Levenshtein for indels.
    """
    if len(a) == len(b):
        d_h = Hamming.distance(a, b, score_cutoff=max_d + 1)
        if d_h <= max_d:
            return d_h
        return Levenshtein.distance(a, b, score_cutoff=max_d + 1)
    else:
        return Levenshtein.distance(a, b, score_cutoff=max_d + 1)


# Minimal BK-tree for approximate search
class BKTree:
    __slots__ = ("root", "dist_fn", "max_children")

    @dataclass
    class Node:
        key: str
        children: dict  # distance -> Node

    def __init__(self, dist_fn):
        self.root = None
        self.dist_fn = dist_fn
        self.max_children = 0

    def add(self, key: str):
        if self.root is None:
            self.root = BKTree.Node(key, {})
            return
        node = self.root
        while True:
            d = self.dist_fn(key, node.key)
            child = node.children.get(d)
            if child is None:
                node.children[d] = BKTree.Node(key, {})
                self.max_children = max(self.max_children, len(node.children))
                return
            node = child

    def query_within(self, key: str, radius: int):
        """Return a key within radius if found (first hit), else None."""
        if self.root is None:
            return None
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = self.dist_fn(key, node.key)
            if d <= radius:
                return node.key
            lo, hi = d - radius, d + radius
            for dd, child in node.children.items():
                if lo <= dd <= hi:
                    stack.append(child)
        return None


# Duplicate marking core
@dataclass
class ReadData:
    name: str
    flag: int
    chrom: str
    start: int
    end: int
    mapq: int
    cigar: str
    seq: str
    qual: str
    strand: str
    cb: str
    umi: str


class Deduper:
    """
    Streaming deduper that:
      - Buckets by (key, start_bin, end_bin) with bin_size = position_tolerance.
      - Uses a BK-tree per bucket to find an existing UMI within umi_ld.
      - Evicts old bins as we stream forward to keep memory flat.
    Decision is made **only on primary alignments**.
    """

    def __init__(
        self,
        umi_ld: int,
        per_cell: bool,
        stranded: bool,
        position_tolerance: int = 10,
    ):
        self.umi_ld = umi_ld
        self.per_cell = per_cell
        self.stranded = stranded
        self.bin = max(1, position_tolerance)

        # key -> start_bin -> end_bin -> {'bktree': BKTree, 'kept_umis': set()}
        self.state = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.active_start_bins = defaultdict(deque)  # key -> deque of start_bins in order

        # stats (primary decisions only)
        self.unique_count = 0
        self.dup_count = 0

    def _key_of(self, chrom: str, cb: str, strand: str):
        if self.per_cell:
            return (chrom, cb, strand) if self.stranded else (chrom, cb)
        return (chrom, strand) if self.stranded else (chrom,)

    def _bins(self, start: int, end: int):
        return start // self.bin, end // self.bin

    def _mk_bktree(self):
        def df(a, b, cutoff=self.umi_ld):
            return _umi_dist(a, b, cutoff)

        return BKTree(df)

    def evict_before(self, chrom_key, current_start_bin: int):
        keep_from = max(0, current_start_bin - 2)
        dq = self.active_start_bins[chrom_key]
        while dq and dq[0] < keep_from:
            old_start_bin = dq.popleft()
            self.state[chrom_key].pop(old_start_bin, None)

    def decide_primary(self, rd: ReadData) -> str:
        """
        Decide duplicate status for a **primary** alignment. Returns "Yes" or "No".
        Exactly one primary per cluster will be "No"; later primaries matching the cluster → "Yes".
        """
        chrom_key = self._key_of(rd.chrom, rd.cb, rd.strand)
        sb, eb = self._bins(rd.start, rd.end)

        # Stream forward eviction
        self.evict_before(chrom_key, sb)

        # Ensure bucket exists
        chrom_dict = self.state[chrom_key]
        if sb not in chrom_dict:
            chrom_dict[sb] = defaultdict(dict)
            self.active_start_bins[chrom_key].append(sb)

        end_dict = chrom_dict[sb]
        bucket = end_dict.get(eb)
        if not bucket:
            bucket = {"bktree": self._mk_bktree(), "kept_umis": set()}
            end_dict[eb] = bucket

        # BK-tree query
        hit = bucket["bktree"].query_within(rd.umi, self.umi_ld)
        if hit is None:
            bucket["bktree"].add(rd.umi)
            bucket["kept_umis"].add(rd.umi)
            self.unique_count += 1
            return "No"
        else:
            self.dup_count += 1
            return "Yes"


# Per-contig worker
def process_region(
    sorted_bam: str,
    temp_bam_path: str,
    region: str,
    umi_ld: int,
    per_cell: bool,
    stranded: bool,
    threads_bgzf: int,
):
    """
    Process a single contig/region:
      - Make duplicate decisions **on primary alignments only**.
      - Propagate decision to secondary/supp if we've already seen the primary in this worker.
      - Write output BAM (coordinate-sorted within this contig).
    """
    deduper = Deduper(
        umi_ld=umi_ld,
        per_cell=per_cell,
        stranded=stranded,
        position_tolerance=10,
    )

    # qname -> bool (True if duplicate) for decisions made on primaries within this region
    qname_dup = {}

    with pysam.AlignmentFile(sorted_bam, "rb") as bam_in:
        hdr = bam_in.header.to_dict()
        hdr.setdefault("HD", {})
        hdr["HD"]["SO"] = "coordinate"
        out_header = pysam.AlignmentHeader.from_dict(hdr)

        with pysam.AlignmentFile(temp_bam_path, "wb", header=out_header, threads=threads_bgzf) as bam_out:
            for read in bam_in.fetch(region):
                if read.is_unmapped:
                    continue

                cb, umi, clean_name = extract_cb_umi(read.query_name)
                if not cb or not umi:
                    # If CB/UMI missing, write through unchanged
                    aln = read.to_string()  # not available; must construct new AlignedSegment
                    # Construct minimal pass-through without DT if you prefer; but since we rely on primaries only,
                    # we will skip here (no CB/UMI -> cannot dedup); write original with no DT/dup change.
                    aln_seg = pysam.AlignedSegment(bam_out.header)
                    aln_seg.query_name = read.query_name
                    aln_seg.flag = read.flag
                    aln_seg.reference_name = read.reference_name
                    aln_seg.reference_start = read.reference_start
                    aln_seg.mapping_quality = read.mapping_quality
                    aln_seg.cigarstring = read.cigarstring
                    aln_seg.query_sequence = read.query_sequence
                    aln_seg.query_qualities = read.query_qualities
                    bam_out.write(aln_seg)
                    continue

                strand = "-" if read.is_reverse else "+"
                rd = ReadData(
                    name=clean_name,
                    flag=read.flag,
                    chrom=read.reference_name,
                    start=read.reference_start,
                    end=read.reference_end,
                    mapq=read.mapping_quality,
                    cigar=read.cigarstring,
                    seq=read.query_sequence,
                    qual=read.qual,
                    strand=strand,
                    cb=cb,
                    umi=umi,
                )

                # Decide only on PRIMARY
                is_primary = not (read.is_secondary or read.is_supplementary)
                if is_primary:
                    dup_str = deduper.decide_primary(rd)  # "Yes" or "No"
                    qname_dup[clean_name] = dup_str == "Yes"
                else:
                    # For secondary/supplementary: reuse decision if known; otherwise leave unchanged
                    dup_str = "Yes" if qname_dup.get(clean_name, False) else None  # None => no DT tag written

                # Build output alignment
                aln = pysam.AlignedSegment(bam_out.header)
                aln.query_name = rd.name  # cleaned QNAME without CB/UMI
                # Set dup flag only if we have a decision (primary or known from prior primary)
                if dup_str is not None and dup_str == "Yes":
                    aln.flag = rd.flag | 0x400
                elif dup_str is not None and dup_str == "No":
                    aln.flag = rd.flag & ~0x400
                else:
                    aln.flag = rd.flag  # unknown for secondaries not yet decided

                aln.reference_name = rd.chrom
                aln.reference_start = rd.start
                aln.mapping_quality = rd.mapq
                aln.cigarstring = rd.cigar
                aln.query_sequence = rd.seq
                aln.query_qualities = pysam.qualitystring_to_array(rd.qual) if isinstance(rd.qual, str) else rd.qual

                # Tags
                aln.set_tag("CB", rd.cb, value_type="Z")
                aln.set_tag("UB", rd.umi, value_type="Z")
                if dup_str is not None:
                    aln.set_tag("DT", dup_str, value_type="Z")  # only when we have a decision

                bam_out.write(aln)

    # Return path; stats are computed after merge on primaries only
    return region, temp_bam_path


# Merge temp BAMs in @SQ order
def merge_in_sq_order(output_bam: str, temp_paths_in_order, template_bam: str, threads_bgzf: int):
    with pysam.AlignmentFile(template_bam, "rb") as template:
        hdr = template.header.to_dict()
        hdr.setdefault("HD", {})
        hdr["HD"]["SO"] = "coordinate"
        out_header = pysam.AlignmentHeader.from_dict(hdr)

        with pysam.AlignmentFile(output_bam, "wb", header=out_header, threads=threads_bgzf) as merged_out:
            for p in temp_paths_in_order:  # same order as @SQ
                with pysam.AlignmentFile(p, "rb") as t:
                    for r in t:
                        merged_out.write(r)


# Per-read (QNAME) stats on primaries only
def compute_final_stats_per_read(bam_path: str, stats_tsv: str, threads: int = 4, primary_only: bool = True):
    """
    Count each read name once:
      - If ANY primary alignment for that QNAME is duplicate -> count as Duplicate.
      - Else -> Unique.
    (Secondary/supplementary are ignored by default.)
    """
    seen = {}
    with pysam.AlignmentFile(bam_path, "rb", threads=threads) as bam:
        for r in bam.fetch(until_eof=True):
            if primary_only and (r.is_secondary or r.is_supplementary):
                continue
            qn = r.query_name
            # Treat either the SAM duplicate bit or DT tag as authority
            is_dup = r.is_duplicate or (r.has_tag("DT") and r.get_tag("DT") == "Yes")
            seen[qn] = seen.get(qn, False) or is_dup

    dups = sum(1 for v in seen.values() if v)
    uniq = len(seen) - dups

    with open(stats_tsv, "w") as fh:
        fh.write("Metric\tValue\n")
        fh.write(f"Unique Reads\t{uniq}\n")
        fh.write(f"Duplicate Reads\t{dups}\n")


def deduplication_parallel(
    sorted_bam: str,
    output_bam: str,
    umi_ld: int = 1,
    per_cell: bool = True,
    threads: int = max(1, mp.cpu_count() // 2),
    stranded: bool = False,
    bgzf_threads_per_writer: int = 4,
):
    """
    Parallel dedup per contig with per-read semantics:
      * Decide duplicates on **primary** alignments only.
      * Exactly one primary per cluster is unique; the rest are duplicates.
      * Write per-contig BAMs with multithreaded BGZF, then merge in @SQ order.
      * Compute **per-read** stats (primaries only).
    """
    # Discover regions in @SQ order
    with pysam.AlignmentFile(sorted_bam, "rb") as bam:
        regions = list(bam.references)

    temp_paths = {}
    logger.info(f"Splitting per contig across {threads} processes ...")

    # Process per region
    work = []
    if threads > 1:
        work = []
        with mp.Pool(processes=threads, maxtasksperchild=32) as pool:
            for region in regions:
                temp_path = f"{output_bam}.{region}.tmp.bam"
                work.append(
                    pool.apply_async(
                        process_region,
                        (
                            sorted_bam,
                            temp_path,
                            region,
                            umi_ld,
                            per_cell,
                            stranded,
                            bgzf_threads_per_writer,
                        ),
                    )
                )

            for w in work:
                region, path = w.get()
                temp_paths[region] = path
    else:
        # Serial path (no Pool)
        for region in regions:
            temp_path = f"{output_bam}.{region}.tmp.bam"
            region_out, path_out = process_region(
                sorted_bam,
                temp_path,
                region,
                umi_ld,
                per_cell,
                stranded,
                bgzf_threads_per_writer,
            )
            temp_paths[region_out] = path_out

    # Merge in @SQ order
    ordered_paths = [temp_paths[r] for r in regions if r in temp_paths]
    merge_in_sq_order(output_bam, ordered_paths, sorted_bam, bgzf_threads_per_writer)

    # Cleanup temps
    for p in ordered_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    # Stats: per read (QNAME), primaries only
    logger.info("Duplicate marking complete, computing per-read stats")
    stats_file = output_bam.replace(".bam", "_stats.tsv")
    compute_final_stats_per_read(output_bam, stats_file, threads=threads, primary_only=True)
    logger.info("Computed per-read stats, indexing final BAM")
