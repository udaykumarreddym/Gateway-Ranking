# LPDG Innovation Hub Selection Challenge 2026

## Machine Learning Track — Gateway Ranking

## 1. Overview

This project ranks gateways for weekly field visits.

For each challenge week, exactly 15 gateways are selected from the available gateway population and ranked from 1 to 15.

The final submission contains:

- `week_start`
- `rank`
- `gateway_id`
- `score`
- `reason`

The final solution combines:

1. a supervised machine-learning fault-risk model, and
2. the supplied 3-sigma anomaly baseline.

The two signals are combined into a 75/25 hybrid ranking.

---

## 2. Challenge Weeks

The challenge contains eight scored weeks:

- 2026-02-02
- 2026-02-09
- 2026-02-16
- 2026-02-23
- 2026-03-02
- 2026-03-09
- 2026-03-16
- 2026-03-23

The final submission therefore contains:

8 weeks × 15 gateways = 120 predictions.

---

## 3. Machine Learning Approach

### Target

The model predicts:

`fault_next_7d`

A positive target represents a confirmed `Fehler behoben` field-visit event requested during the seven-day period beginning at the prediction cutoff.

The event timing uses `requested_on`.

The target window is:

`[cutoff, cutoff + 7 days)`

Repeated confirmed repair events within the same gateway-week are collapsed into a binary target.

`Kein Fehler gefunden` and `Kein Zugang` are not treated as positive fault events.

---

## 4. Features

The historical training dataset contains:

- 7,304 gateway-week observations
- 332 gateways
- 22 historical prediction weeks
- 91 positive gateway-week observations
- 7,213 negative gateway-week observations
- 1.246% positive rate

The final model uses 51 features.

Feature groups include:

- gateway metadata
- telemetry behavior
- meter-read behavior
- offline-duration statistics
- disconnection statistics
- reboot statistics
- transmission/load-related telemetry retained during feature selection

Telemetry features include rolling-window statistics and trends calculated from historical observations.

Meter-read features include recent success rates and trends.

---

## 5. Cutoff Safety

Feature construction follows a strict temporal cutoff.

For a prediction week beginning on Monday:

- telemetry at or after the Monday cutoff is not used as a feature;
- telemetry lookback features use historical observations before the cutoff;
- meter-read features are shifted so that only previously completed meter-read periods are available;
- future field visits are not used as prediction features;
- future challenge outcomes are not used as prediction features.

The challenge feature matrix contains no target column.

This prevents future information from being incorporated into the prediction features.

---

## 6. Model

The machine-learning component uses scikit-learn's:

`HistGradientBoostingClassifier`

The final training configuration uses:

- `max_iter=300`
- `learning_rate=0.05`
- `max_leaf_nodes=15`
- `l2_regularization=2.0`
- `random_state=42`

Because the positive class is rare, positive examples receive a sample weight of 2.0 while negative examples receive a weight of 1.0.

The trained model is stored at:

`models/gateway_fault_model_weight2.joblib`

---

## 7. 3-Sigma Baseline

The supplied 3-sigma baseline is used as the second ranking signal.

It evaluates the following telemetry metrics:

- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`

For each gateway, statistics are calculated from the preceding 28 days.

For the most recent seven days, observations exceeding:

`mean + 3 × std`

are counted as anomalies.

The resulting anomaly count is used as the baseline score.

The supplied baseline implementation remains in the repository as:

`baseline_3sigma.py`

The corresponding validator remains as:

`validate_submission.py`

---

## 8. Hybrid Ranking

The final ranking combines the machine-learning and 3-sigma signals using within-week percentile ranks:

```text
Hybrid Score =
    0.75 × ML percentile
    + 0.25 × 3-sigma baseline percentile
```
