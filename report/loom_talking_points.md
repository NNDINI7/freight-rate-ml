# Loom outline (2-3 min) — talk over your screen, not a script to read verbatim

**0:00-0:25 — Key findings from exploring the data**
- posted_rate is driven mostly by distance (corr 0.91); quote_signal adds a bit more, mainly through distance × quote_signal.
- market_index is basically a shared daily market factor (tiny within-date spread); quote_signal is much noisier per shipment.
- Small cluster of very high $/mile short-haul loads (~0.7%) — likely legit spot/expedite premiums, not errors.

**0:25-0:55 — Data-quality issues and how you addressed them**
- Negative weight values (sign-flip typos, ~0.6%) → fixed with abs().
- Missing weight (~0.6%) and missing market_index (~1-2%) → weight: median impute; market_index: same-date average (justified since it's date-level, not per-load).
- December chart inputs missing market_index/quote_signal entirely → market_index recovered exactly from train+validation combined; quote_signal filled with the lane+equipment historical median (call out this is an approximation).

**0:55-1:30 — Reasoning behind the chosen model**
- LightGBM on log(posted_rate): handles the mixed categorical/numeric features and nonlinear interactions (e.g. distance × quote_signal) without manual encoding gymnastics; fast to train/iterate on 48k rows.
- Trained in log space since rates are right-skewed and errors should scale proportionally.

**1:30-2:10 — Training and validation approach**
- Time-based split (last 15% of dates held out), not random — validation.csv is entirely future dates, so the holdout mirrors that.
- Holdout MAE ~$119, MAPE ~5.4%, 97% of predictions within 10%.
- Final model refit on 100% of labeled data before scoring validation.csv (best_iteration from the holdout run, scaled up slightly for the extra data).

**2:10-2:45 — Code walkthrough**
- src/features.py: shared cleaning + feature engineering (one source of truth for train/predict).
- src/train.py: time-based split, LightGBM with early stopping, saves metrics + model.
- src/predict.py: refits on all data, produces validation_predictions.csv and fills december_chart_inputs.csv.
- src/score.py: the provided scorer — validates format and renders the chart.

Keep it conversational — walk through the actual files/terminal rather than reading this.
