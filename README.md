# PPP Risk Analytics — Streamlit Customer Demo

A three-page customer-facing front end for PPP loan review prioritization:

1. **Executive Overview** — portfolio KPIs, score distribution, loan-vs-score scatter, industry concentration, US map, highest-priority queue.
2. **Anomaly Explorer** — searchable/filtered review queue, selected-loan details, reason codes and peer comparisons.
3. **Model Insights** — score/tier behavior, population comparisons, common review signals and integration explanation.

## Why there are two scoring modes

The supplied capstone notebook's final prioritization is **XGBoost-anchored** (`risk_score` is based on XGBoost percentile rank) with Isolation Forest retained as an informational `if_anomaly` signal. The supplied PPP ZIP contains the loan table, but not the separate fraud-case CSV the notebook needs to recreate the supervised labels/models from scratch.

Therefore this package works immediately with an **Isolation Forest demo mode**, and it also accepts a notebook-exported scored table without changing the dashboard.

> Dashboard language intentionally says **risk / priority / unusual**, not “confirmed fraud.” A model flag is a review-prioritization signal.

## Quick start with the included demo database

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The included `data/ppp_demo.db` is a scored sample from the uploaded PPP dataset so the app opens immediately.

## Rebuild the demo database from the uploaded PPP ZIP

```bash
python scripts/build_demo_db.py --zip "/path/to/ppp_over_150k.zip" --rows 30000
streamlit run app.py
```

## Connect the capstone notebook's final model

After `compare_df` exists in the notebook, run the logic in `scripts/export_notebook_scores.py`. It writes `ppp_dashboard_scored.csv` with the final `risk_score`, `risk_tier`, `xgb_proba`, `iso_score`, and `if_anomaly` fields.

For production/demo integration, load that scored table into SQLite/Postgres/Snowflake and point the front end's data loader to it. The UI contract only requires one scored-loan table.

### Minimum scored-table contract

- LoanNumber
- BorrowerName
- BorrowerState
- NAICSCode / industry
- CurrentApprovalAmount
- JobsReported
- ForgivenessAmount / NotForgivenAmount
- risk_score (0–1)
- risk_tier (High / Medium / Low)
- if_anomaly
- primary_risk_driver
- secondary_risk_driver

Optional notebook fields: `xgb_proba`, `iso_score`, `xgb_rank`, `iso_rank`, `y_true`.

## Suggested customer demo flow

1. Open **Executive Overview** and describe portfolio-level monitoring.
2. Filter to a state/industry and show every visual responding.
3. Open **Anomaly Explorer**, select a high-priority borrower, and explain the peer comparison / reason codes.
4. Open **Model Insights** and emphasize transparency plus the limitation that prioritization is not a fraud determination.

## Database path

Default: `data/ppp_demo.db`, table `loans_scored`.

The database layer is deliberately separated from the UI (`src/data.py`) so you can later replace SQLite with Postgres, Snowflake, Redshift, etc. without redesigning the dashboard.
