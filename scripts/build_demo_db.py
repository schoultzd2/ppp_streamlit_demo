from __future__ import annotations
from pathlib import Path
import argparse, sqlite3, zipfile
import numpy as np
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring import score_isolation_forest

USECOLS = ["LoanNumber","DateApproved","BorrowerName","BorrowerCity","BorrowerState","NAICSCode","BusinessType","OriginatingLender","CurrentApprovalAmount","JobsReported","ForgivenessAmount","NotForgivenAmount","ForgivenPercentage","PROCEED_Per_Job"]

def sampled_read(zip_path: Path, target=30000, seed=42):
    rng=np.random.default_rng(seed); parts=[]
    with zipfile.ZipFile(zip_path) as z:
        name=[n for n in z.namelist() if n.lower().endswith('.csv')][0]
        with z.open(name) as f:
            for chunk in pd.read_csv(f, usecols=lambda c:c in USECOLS, chunksize=100000, low_memory=False):
                n=min(max(1,int(target/12)),len(chunk))
                parts.append(chunk.sample(n=n, random_state=int(rng.integers(1_000_000))))
                if sum(len(x) for x in parts) >= target*2: break
    df=pd.concat(parts,ignore_index=True)
    if len(df)>target: df=df.sample(target,random_state=seed).reset_index(drop=True)
    return df

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--out", default=str(ROOT/"data"/"ppp_demo.db"))
    p.add_argument("--rows", type=int, default=30000)
    args=p.parse_args()
    raw=sampled_read(Path(args.zip),args.rows)
    scored=score_isolation_forest(raw)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(args.out) as conn:
        scored.to_sql("loans_scored",conn,if_exists="replace",index=True,index_label="row_id")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_risk ON loans_scored(risk_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON loans_scored(BorrowerState)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON loans_scored(risk_tier)")
    print(f"Wrote {len(scored):,} scored rows to {args.out}")

if __name__=="__main__": main()
