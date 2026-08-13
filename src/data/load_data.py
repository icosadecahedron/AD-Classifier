"""
Data loading module.

Supports two sources, controlled by config.data.source:
  - "geo_series_matrix": parses a real GEO series matrix file (e.g. GSE33000).
    Point config.data.geo_path at the downloaded file from:
    ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE33nnn/GSE33000/matrix/
  - "synthetic": generates a realistic stand-in dataset with the same shape
    (samples x genes, binary AD/control label) so the pipeline is fully
    runnable without network access. Swap to "geo_series_matrix" with zero
    other code changes once you have the real file.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _generate_synthetic_expression_data(
    n_samples: int, n_genes: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)

    labels = rng.binomial(1, 0.5, size=n_samples)  # 1 = AD, 0 = control
    n_signal_genes = max(20, int(n_genes * 0.05))

    base_expression = rng.normal(loc=8.0, scale=1.5, size=(n_samples, n_genes))

    # Inject a real, learnable signal into a subset of "AD risk-like" genes
    signal_idx = rng.choice(n_genes, size=n_signal_genes, replace=False)
    effect_size = rng.normal(loc=1.2, scale=0.4, size=n_signal_genes)
    for i, gene_idx in enumerate(signal_idx):
        base_expression[labels == 1, gene_idx] += effect_size[i]

    gene_names = [f"GENE_{i:05d}" for i in range(n_genes)]
    # Give the known AD-associated genes recognizable names among the signal genes,
    # purely for a more realistic/interpretable demo downstream (SHAP output, etc.)
    known_ad_genes = ["APOE", "BIN1", "CLU", "PICALM", "TREM2", "MAPT", "ABCA7", "SORL1"]
    for i, gene_idx in enumerate(signal_idx[: len(known_ad_genes)]):
        gene_names[gene_idx] = known_ad_genes[i]

    expr_df = pd.DataFrame(base_expression, columns=gene_names)
    expr_df.insert(0, "sample_id", [f"SAMPLE_{i:04d}" for i in range(n_samples)])

    label_series = pd.Series(labels, name="diagnosis")
    return expr_df, label_series


def _load_geo_series_matrix(path: str, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"GEO series matrix not found at {path}. Download it from GEO and "
            f"place it at this path, or set data.source back to 'synthetic'."
        )

    with open(path_obj) as f:
        lines = f.readlines()

    def _parse_geo_row(line: str) -> list[str]:
        """Split a '!Sample_xxx\t"val1"\t"val2"...' line into stripped values."""
        parts = line.rstrip("\n").split("\t")[1:]
        return [p.strip('"') for p in parts]

    gsm_line = next(l for l in lines if l.startswith("!Sample_geo_accession"))
    gsm_order = _parse_geo_row(gsm_line)

    disease_status_line = next(
        l for l in lines
        if l.startswith("!Sample_characteristics_ch2") and "disease status" in l.lower()
    )
    raw_statuses = _parse_geo_row(disease_status_line)
    statuses = [v.split(": ", 1)[1] if ": " in v else v for v in raw_statuses]
    gsm_to_status = dict(zip(gsm_order, statuses))

    keep_gsms = {
        gsm for gsm, status in gsm_to_status.items()
        if status in ("Alzheimer's disease", "non-demented")
    }
    label_map = {"Alzheimer's disease": 1, "non-demented": 0}

    start = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    end = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    table_lines = lines[start + 1 : end]

    from io import StringIO

    raw_table = pd.read_csv(StringIO("".join(table_lines)), sep="\t", index_col=0)
    kept_columns = [c for c in raw_table.columns if c in keep_gsms]
    filtered_table = raw_table[kept_columns]

    expr_df = filtered_table.T
    expr_df.index.name = "sample_id"
    expr_df = expr_df.reset_index().rename(columns={"sample_id": "sample_id"})
    expr_df["sample_id"] = kept_columns

    labels = pd.Series(
        [label_map[gsm_to_status[gsm]] for gsm in kept_columns], name=label_col
    )

    logger.info(
        "Loaded real GSE33000 data: %d samples (dropped %d Huntington's disease samples), %d probes",
        len(kept_columns), len(gsm_order) - len(keep_gsms), filtered_table.shape[0],
    )
    return expr_df, labels



def load_dataset(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Main entry point: returns (expression_df, labels) per config.data.source."""
    data_cfg = config["data"]

    if data_cfg["source"] == "synthetic":
        logger.info("Loading SYNTHETIC gene expression dataset (no network access in this environment).")
        return _generate_synthetic_expression_data(
            n_samples=data_cfg["n_samples"],
            n_genes=data_cfg["n_genes"],
            seed=data_cfg["random_seed"],
        )
    elif data_cfg["source"] == "geo_series_matrix":
        logger.info("Loading real GEO series matrix from %s", data_cfg["geo_path"])
        return _load_geo_series_matrix(data_cfg["geo_path"], data_cfg["label_col"])
    else:
        raise ValueError(f"Unknown data source: {data_cfg['source']}")
