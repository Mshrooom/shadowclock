# DATA = folder written by notebooks/kaggle_render_real.ipynb (see README, "Regenerate the dataset")
DATA ?= data_real

.PHONY: test results figures map world web all

test:
	python -m pytest -q

results:
	python scripts/build_city_results.py --data $(DATA) --pairs results/pairs_real.jsonl --preds results/preds_baseline_auto_real.jsonl --out results

figures:
	python scripts/make_figures.py --results results --out results/figures

world:
	python scripts/build_world_data.py --data $(DATA)

map:
	python scripts/make_map.py --results results --samples data/samples --out docs/map.html

web:
	cd web && npm install && npm run build
