# Limitations and open items

1. Simulated, not observed. No real aerial photographs are used. Sundial's six NAIP images are the obvious real-image
   check and have not been run.
2. Heights. About half of the OSM heights are assumed defaults.
3. Sample size. Twelve scenes, about 30 distinct pairs per city, one date range per scene. Per-city intervals are wide.
4. Models. Three small open models; two always answer "A". No frontier model has been tested. Do not generalise.
5. Unexplained: the baseline's R2 accuracy is lower than its R1 accuracy on the OSM scenes.
6. Recorded, not reproducible from this repository: the synthetic-set model numbers in `results/synthetic_results.json`
   were copied from Kaggle printouts; raw predictions are not stored. The Qwen2.5-VL-3B pair count and interval were not
   recorded. The latitudes used in the synthetic set are not recorded in this repository.
7. No human baseline yet. The quiz in the viewer can collect one, but a few answers are a usability check.
8. Prior work. Close predecessors exist; novelty has not been established.
9. Provenance. `pipeline/geom.py`, `pipeline/generate_dataset.py` and `pipeline/hemisphere_flip.py` were written earlier in
   this project and are included as they were; they have tests only through the end-to-end toy run.
