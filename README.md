# Alzheimer's Disease Classification Pipeline & Serving API

An end-to-end ML pipeline that classifies Alzheimer's disease (AD) status from
brain gene expression data, with differential-expression-based feature
selection, multi-model benchmarking with experiment tracking, and a
Dockerized FastAPI serving layer with SHAP explanations.

Real AD gene-expression cohorts (e.g. ROSMAP) require a Data Use Agreement
and Synapse access. This project is built to run against **open GEO data**
instead — specifically [GSE33000](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000),
AD vs. control prefrontal cortex expression (~600 samples), originally
published in:

> Zhang, B. et al. "Integrated systems approach identifies genetic nodes and
> networks in late-onset Alzheimer's disease." *Cell* 153, 707–720 (2013).

with zero application process for the data itself — just a download.

By default, `configs/config.yaml` is set to `data.source: synthetic`, which
generates a realistic stand-in dataset (same shape and structure as a real
microarray/RNA-seq cohort, with a genuine learnable signal injected into a
subset of genes named after real AD risk genes) so the **entire pipeline
runs immediately with no setup**. Swap to `data.source: geo_series_matrix`
and point `data.geo_path` at a downloaded GEO series matrix file to run on
real data with zero other code changes.

> **Note on metrics**: numbers produced against the synthetic dataset (e.g.
> AUC = 1.0) reflect that the injected signal is deliberately learnable —
> they demonstrate the pipeline works end-to-end, not real-world AD
> classification performance. Expect meaningfully lower, more realistic
> numbers on the real GEO dataset; report those, not the synthetic ones.

## Architecture

```
GEO / synthetic data → preprocessing (log transform, variance filter)
                     → differential expression (Welch's t-test + BH-FDR)
                     → feature selection (significant, effect-size-filtered genes)
                     → model benchmarking (LogReg, RandomForest, XGBoost)
                       with MLflow experiment tracking + stratified CV
                     → best model persisted (joblib)
                     → FastAPI serving layer (/predict, /features, /health)
                       with SHAP-based explanations
```

## Project structure

```
ad-classifier/
├── configs/config.yaml       # all pipeline parameters
├── src/
│   ├── data/                 # loading + preprocessing
│   ├── features/             # differential expression feature selection
│   ├── models/                # training + MLflow tracking
│   ├── api/                  # FastAPI serving layer
│   └── pipeline.py           # main entry point
├── tests/                    # pytest unit + API tests
├── Dockerfile / docker-compose.yml
├── .github/workflows/ci.yml  # lint + test + docker build on every push
└── Makefile                  # make train / make test / make serve
```

## Running it

```bash
make install     # pip install -r requirements.txt
make train       # runs the full pipeline, trains + logs models via MLflow
make test        # runs the pytest suite
make serve       # starts the FastAPI server on :8000
```

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gene_expression": {"APOE": 9.4, "BIN1": 8.1}}'
```

View tracked experiments:

```bash
mlflow ui --backend-store-uri mlruns
```

## Development environment note

The project has now been run and validated locally in a networked macOS
environment using Python 3.11. The full dependency stack, including
`xgboost`, `mlflow`, `shap`, `fastapi`, and `pytest`, was installed
successfully.

The synthetic-data training pipeline completed end to end, benchmarked
Logistic Regression, Random Forest, and XGBoost, tracked experiments with
MLflow using a local SQLite backend, and saved the best model to
`models/best_model.joblib`.

The FastAPI service was started and its `/health`, `/features`, and `/predict`
endpoints were tested successfully. All 16 automated tests pass.

Docker and GitHub Actions configuration are included in the repository, but
the Docker image has not yet been successfully built or tested on the current
development machine.

Docker testing will be coming as soon as I get my next energy drink shipment ;)