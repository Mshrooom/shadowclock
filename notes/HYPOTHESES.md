# Frozen hypotheses and decision thresholds

**Freeze rule:** commit this file (git commit, or email it to yourself) BEFORE running any model on the
full data. `evaluate_hypotheses.py` applies these rules mechanically; if you change a threshold after seeing
results, say so in the paper. The thresholds are proposals chosen before any model result existed; they are
conventions, not physical constants.

Data: n = 464 distinct pairs (+ 464 A/B-swapped twins), gaps 1/2/4 h, sun elevation <= 70 deg, 8 scenes,
4 latitudes (two per hemisphere). Regimes: R0 = images only, R1 = + hemisphere, R2 = + latitude/longitude/date.
Chance level for ordering is 50%. Reference "no-skill" elapsed-time guess = the best constant (2 h): MAE ~0.97 h
on the 928-pair set (hand arithmetic from the gap counts; the script prints the exact value).

**Gate.** A (model, regime) with > 10% unparseable answers, or missing answers, is NOT SCORED (reported as a
format failure, not as a hypothesis result).

| id | claim | SUPPORTED if | FALSIFIED if |
|---|---|---|---|
| H1 | Without hemisphere information (R0) a VLM falls back on a northern-hemisphere prior | north acc >= 60% and north - south >= 20 pts | south acc >= 60%, or abs(north - south) < 10 pts (no prior detectable) |
| H2 | Given the hemisphere (R1), ordering 1 h-apart pairs is above chance but clearly below the analytic ceiling | roll-up: >= half of scored models score PARTIAL (55% < acc@1h < 90%) | all scored models score SOLVED (acc@1h >= 90%) |
| H3 | Elapsed-time estimates are barely better than a constant guess, even with latitude and date (R2) | per model: MAE_R1 >= 0.75 x no-skill MAE and MAE_R2 >= 0.5 h; roll-up >= half of models | per model: MAE_R2 < 0.5 h; roll-up all models |
| H4 | Answers depend on image position, not only on the shadows | per model: swap consistency (R1) < 90%; roll-up >= half of models | per model: >= 95%; roll-up all models |
| H5 | (dropped) RS-specific VLMs do no better than general VLMs | not testable with the models run here | - |

Per-model class for H2: SOLVED (acc@1h >= 90), CHANCE (<= 55), PARTIAL (otherwise). Models at CHANCE count
toward "not solved" but are reported separately as the stronger "blindness" outcome.

**Perception probe (diagnosis, not a hypothesis).** Single-image 8-way shadow-direction question.
Perception OK if within-one-sector accuracy >= 70% (chance 37.5%). A model that is not SOLVED in R1 but passes
the probe has a reasoning failure; one that fails the probe has a perception failure.

**Known limits of these rules.** Thresholds (55/60/90/0.75/0.5/70) are arbitrary conventions. Only 8 scenes
with images reused across pairs: Wilson intervals assume independence (reported on distinct pairs) and the
scene-bootstrap with 8 clusters is crude. Synthetic, simple scenes; no real-image validation yet. The "round
number answers" sub-claim from the earlier draft was dropped: the true gaps are themselves round numbers.

## Correction logged after the first pilots (thresholds unchanged)
The probe "chance" levels printed at first (12.5% exact, 37.5% within one sector) assume equally likely shadow sectors.
They are not: with this dataset's sun positions east/west shadows are the most common (about 19% each by my
calculation for the render grid), so always answering one fixed direction already scores about 19% exact and about 40%
within one sector. The evaluator now also prints the best constant-answer level computed from the actual probe set and the
model's most frequent answer. The frozen decision rule (perception OK only if within-one-sector >= 70%) is unchanged.
Also: with small n the H2 rule can label a guessing model PARTIAL, and the H4 rule (< 90%) is met by random guessing; read
both together with the descriptive overall-accuracy and A-rate intervals the evaluator prints.
