.PHONY: fetch-datasets build-golden-dataset

# Fetching datasets (sources)
fetch-datasets:
	uv run python -m src.golden_dataset.datasets_handlers.fetch_datasets

# Assemble all source parquets into the single golden arguments pool
build-golden-dataset:
	uv run python -m src.golden_dataset.datasets_handlers.assemble