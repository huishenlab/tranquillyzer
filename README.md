# Tranquillyzer

[![codecov](https://codecov.io/github/AyushSemwal/tranquillyzer/graph/badge.svg?token=QS4IK3UZRN)](https://codecov.io/github/AyushSemwal/tranquillyzer)

**Tranquillyzer** (**TRAN**script **QU**antification **I**n **L**ong reads-ana**LYZER**), is a flexible,
architecture-aware deep learning framework for processing long-read single-cell RNA-seq (scRNA-seq) data. It employs a
hybrid neural network architecture and a global, context-aware design that enables the precise identification of
structural elements. In addition to supporting established single-cell protocols, Tranquillyzer accommodates custom
library formats through rapid, one-time model training on user-defined label schemas. Model training for both
established and custom protocols can typically be completed within a few hours on standard GPUs.

For a detailed description of the framework, benchmarking results, and application to real datasets, please refer to the
[preprint](https://www.biorxiv.org/content/10.1101/2025.07.25.666829v1).

# Citation

### bioRxiv

```
Tranquillyzer: A Flexible Neural Network Framework for Structural Annotation and
Demultiplexing of Long-Read Transcriptomes. Ayush Semwal, Jacob Morrison, Ian
Beddows, Theron Palmer, Mary F. Majewski, H. Josh Jang, Benjamin K. Johnson, Hui
Shen. bioRxiv 2025.07.25.666829; doi: https://doi.org/10.1101/2025.07.25.666829.
```

# Overview

Tranquillyzer includes several steps to process reads from a raw basecalled FASTA/FASTQ file to a deduplicated BAM to
creating a feature counts matrix. First, Tranquillyzer preprocesses the reads to collect metadata on the reads and sort
them into bins of similar lengths to ease downstream processing. Next, Tranquillyzer annotates the reads using a hybrid
neural network architecture to identify each structural element in a read. It also demultiplexes reads to their
respective cells at this time. After annotating and demultiplexing, the reads are aligned and PCR duplicate marked. The
BAM output from this step can then be used to determine feature counts matrices. Tranquillyzer also provides a variety
of associated functionality including visualizing annotated reads and quality control metrics, training models for new
sequencing architectures or to improve the annotation capability, and the ability to simulate reads for use in model
training. A more detailed overview of Tranquillyzer can be
[found in the online documentation](https://huishenlab.github.io/tranquillyzer/).

# Quick Start and General Usage

Documentation for Tranquillyzer: <https://huishenlab.github.io/tranquillyzer/>.

For a guide to getting started with Tranquillyzer, see the
[Quick Start guide](https://huishenlab.github.io/tranquillyzer/webpages/quick_start.html). For more detailed notes on
using Tranquillyzer, see the [Usage page](https://huishenlab.github.io/tranquillyzer/webpages/usage.html).

# Installation

Tranquillyzer is available through a variety of methods. See the
[Installation](https://huishenlab.github.io/tranquillyzer/webpages/install.html) page for details.

# Issues

Issues can be opened on GitHub: <https://github.com/huishenlab/tranquillyzer/issues>.

# Acknowledgements

This work was supported by Van Andel Research Institute start-up funding and National Institutes of Health [UM1DA058219]
to Hui Shen. Computation for the work described in this paper was supported by the High-Performance Cluster and Cloud
Computing (HPC3) Resource at the Van Andel Research Institute.

# Disclosure of AI-Assisted Development

During the preparation of this work the authors used ChatGPT (OpenAI) and AI-based software development tools to assist
with language refinement and code debugging. The authors developed and verified all scientific content, interpretation
and conclusions.
