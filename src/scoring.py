from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

FEATURES = [
    "CurrentApprovalAmount", "JobsReported", "ForgivenessAmount",
    "NotForgivenAmount", "ForgivenPercentage", "PROCEED_Per_Job"
]

SECTOR_NAMES = {
    "11":"Agriculture", "21":"Mining", "22":"Utilities", "23":"Construction",
    "31":"Manufacturing", "32":"Manufacturing", "33":"Manufacturing",
    "42":"Wholesale Trade", "44":"Retail Trade", "45":"Retail Trade",
    "48":"Transportation", "49":"Transportation", "51":"Information",
    "52":"Finance & Insurance", "53":"Real Estate", "54":"Professional Services",
    "55":"Management", "56":"Administrative Services", "61":"Education",
    "62":"Health Care", "71":"Arts & Entertainment", "72":"Accommodation & Food",
    "81":"Other Services", "92":"Public Administration"
}

def enrich_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = ["CurrentApprovalAmount","JobsReported","ForgivenessAmount","NotForgivenAmount","ForgivenPercentage","PROCEED_Per_Job"]
    for c in numeric:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "PROCEED_Per_Job" not in out or out["PROCEED_Per_Job"].isna().all():
        out["PROCEED_Per_Job"] = out["CurrentApprovalAmount"] / out["JobsReported"].replace(0, np.nan)
    out["amount_per_job"] = out["CurrentApprovalAmount"] / out["JobsReported"].replace(0, np.nan)
    out["amount_per_job"] = out["amount_per_job"].replace([np.inf,-np.inf], np.nan)
    out["naics_sector"] = out.get("NAICSCode", pd.Series(index=out.index, dtype="object")).astype("Int64").astype(str).str[:2]
    out["industry"] = out["naics_sector"].map(SECTOR_NAMES).fillna("Other / Unknown")
    if "BorrowerState" in out:
        out["BorrowerState"] = out["BorrowerState"].fillna("UNK").astype(str).str.upper()
    return out

def score_isolation_forest(df: pd.DataFrame, contamination: float = 0.01, random_state: int = 42) -> pd.DataFrame:
    """Demo-mode unsupervised scorer. Higher score = more unusual."""
    out = enrich_base(df)
    feats = [c for c in FEATURES if c in out.columns]
    X = out[feats].copy()
    for c in feats:
        X[c] = X[c].replace([np.inf,-np.inf], np.nan)
        X[c] = X[c].fillna(X[c].median())
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(Xs)
    raw = -model.decision_function(Xs)
    lo, hi = np.nanmin(raw), np.nanmax(raw)
    norm = (raw - lo) / (hi - lo + 1e-12)
    out["iso_score"] = norm
    out["iso_rank"] = pd.Series(norm, index=out.index).rank(pct=True)
    out["risk_score"] = out["iso_rank"]
    out["if_anomaly"] = out["iso_rank"] >= 0.99
    pct = out["risk_score"].rank(pct=True)
    out["risk_tier"] = np.select([pct >= .99, pct >= .95], ["High", "Medium"], default="Low")
    out["scoring_mode"] = "Isolation Forest demo"
    out = add_explanations(out)
    return out

def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # peer benchmarks by broad industry; fall back to portfolio median
    grp = out.groupby("industry", dropna=False)
    out["peer_median_loan"] = grp["CurrentApprovalAmount"].transform("median")
    out["peer_median_jobs"] = grp["JobsReported"].transform("median")
    out["peer_median_apj"] = grp["amount_per_job"].transform("median")
    portfolio_loan = out["CurrentApprovalAmount"].median()
    portfolio_apj = out["amount_per_job"].median()
    out["peer_median_loan"] = out["peer_median_loan"].fillna(portfolio_loan)
    out["peer_median_apj"] = out["peer_median_apj"].fillna(portfolio_apj)

    out["industry_loan_percentile"] = grp["CurrentApprovalAmount"].rank(pct=True)
    out["industry_apj_percentile"] = grp["amount_per_job"].rank(pct=True)

    def reasons(r):
        vals = []
        if pd.notna(r.get("industry_apj_percentile")) and r["industry_apj_percentile"] >= .95:
            vals.append("High loan dollars per reported job")
        if pd.notna(r.get("industry_loan_percentile")) and r["industry_loan_percentile"] >= .95:
            vals.append("Loan amount is high versus industry peers")
        if pd.notna(r.get("JobsReported")) and pd.notna(r.get("peer_median_jobs")) and r["JobsReported"] < max(2, .25*r["peer_median_jobs"]):
            vals.append("Low reported jobs versus industry peers")
        if pd.notna(r.get("NotForgivenAmount")) and r["NotForgivenAmount"] > 0:
            vals.append("Positive not-forgiven balance")
        if not vals:
            vals.append("Unusual multivariate feature combination")
        return vals[:2]
    rr = out.apply(reasons, axis=1)
    out["primary_risk_driver"] = rr.str[0]
    out["secondary_risk_driver"] = rr.apply(lambda x: x[1] if len(x)>1 else "")
    return out

def ensure_scored(df: pd.DataFrame) -> pd.DataFrame:
    """Accept notebook-exported scores when present; otherwise create demo scores."""
    if "risk_score" in df.columns and "risk_tier" in df.columns:
        out = enrich_base(df)
        if "if_anomaly" not in out:
            out["if_anomaly"] = out.get("iso_rank", 0) >= .99
        if "scoring_mode" not in out:
            out["scoring_mode"] = "Notebook-exported hybrid score"
        if "primary_risk_driver" not in out:
            out = add_explanations(out)
        return out
    return score_isolation_forest(df)
