# Modeling Decisions

## 1. Gateway universe

The training dataset starts from all 332 gateways rather than only gateways with confirmed historical faults.

Reason:

A model must rank the complete gateway population, including gateways for which no confirmed repair event was observed.

---

## 2. Positive target

A positive event is based on a field visit with:

`outcome = Fehler behoben`

Reason:

This is the strongest available indication that a real fault was found and repaired.

---

## 3. Event timing

`requested_on` is used for the prediction target rather than `visited_on`.

Reason:

The request date is closer to when the fault became observable. Physical visit dates can occur several days later.

---

## 4. Weekly target

The target is defined over:

`[prediction cutoff, prediction cutoff + 7 days)`

Multiple events within the same gateway-week are collapsed to one positive label.

---

## 5. Temporal validation

Forward-in-time validation was used instead of a random train/test split.

Reason:

Random splitting could allow future behavior to influence training and would provide an overly optimistic estimate of deployment performance.

---

## 6. Feature cutoff

Prediction features use information strictly before the prediction week's cutoff.

Telemetry uses the previous 28 days.

Meter-read features are shifted so the current week's result is not used at Monday prediction time.

---

## 7. Feature reduction

The initial feature set was reduced before model training.

Reason:

Only 91 positive observations were available. A very large feature space relative to the number of positives increases the risk of unstable or overly complex models.

---

## 8. Model

HistGradientBoostingClassifier was selected as the available tree-based model.

XGBoost was not available in the execution environment.

The final configuration uses:

- `max_iter = 300`
- `learning_rate = 0.05`
- `max_leaf_nodes = 15`
- `l2_regularization = 2.0`
- `random_state = 42`

Positive training examples receive sample weight 2.0 and negative examples receive weight 1.0.

---

## 9. Baseline

The supplied 3-sigma baseline was reproduced independently before using it in the hybrid.

Exact reproduction achieved:

`120/120` selections and ranks.

This provides confidence that the comparison is against the actual challenge baseline rather than an approximation.

---

## 10. Hybrid ranking

The final ranking combines:

`75% ML percentile + 25% baseline percentile`

Reason:

The validation analysis showed complementary behavior.

On the 12-week forward validation period (2025-11-10 through 2026-01-26):

- 3-sigma baseline: 21 faults caught, total cost €86,400
- HGB model: 21 faults caught, total cost €86,400
- 75/25 hybrid: 27 faults caught, total cost €82,800

The hybrid therefore showed lower validation cost than either individual component during this validation period.

The ML model tends to capture sustained behavioral deterioration, while the 3-sigma method responds strongly to acute spikes.

---

## 11. Missing baseline scores

Some gateway-weeks have no telemetry observations in the baseline lookback window.

For the complete hybrid ranking matrix, these are assigned a baseline score of zero.

Reason:

No observed 3-sigma anomaly provides no positive evidence from that baseline component. This allows every gateway-week to participate in the hybrid ranking.

---

## 12. Ranking score

The submitted score is the hybrid ranking score rather than a calibrated probability.

Reason:

The challenge requires prioritization/ranking of exactly 15 gateways per week. The score is therefore intended primarily for ordering.

---

## 13. Explanation generation

Reasons are generated only after the ranking is finalized.

Reason:

The explanation stage must not influence gateway selection.

Reasons are based on actual observed feature evidence and the model/baseline signals.

---

## 14. Validation caveat

The 75/25 hybrid weight was selected after examining the available validation results.

It should therefore not be presented as an independently validated hyperparameter.

A future production version should reserve a final untouched temporal holdout for selecting the hybrid weight.