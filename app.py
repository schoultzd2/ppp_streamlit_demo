from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data import load_sqlite, load_csv_or_zip
from src.scoring import ensure_scored
from src.ui import CSS, money, number

st.set_page_config(page_title="PPP Risk Analytics", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

ROOT = Path(__file__).parent
DEFAULT_DB = ROOT / "data" / "ppp_demo.db"
DEFAULT_RAW = Path(os.getenv("PPP_DATA_PATH", ROOT / "data" / "ppp_over_150k.zip"))

@st.cache_data(show_spinner=False)
def get_data():
    if DEFAULT_DB.exists():
        df = load_sqlite(DEFAULT_DB)
    elif DEFAULT_RAW.exists():
        df = ensure_scored(load_csv_or_zip(DEFAULT_RAW, nrows=50000))
    else:
        return pd.DataFrame()
    for c in ["CurrentApprovalAmount","JobsReported","ForgivenessAmount","NotForgivenAmount","ForgivenPercentage","risk_score","iso_score","xgb_proba","amount_per_job","peer_median_loan","industry_loan_percentile","industry_apj_percentile"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = get_data()

st.sidebar.markdown("## ◈ PPP Risk Analytics")
st.sidebar.caption("ML-enabled review prioritization")
page = st.sidebar.radio("Navigate", ["Executive Overview", "Anomaly Explorer", "Model Insights"], label_visibility="collapsed")
st.sidebar.markdown("---")
if len(df):
    states = sorted([s for s in df["BorrowerState"].dropna().unique() if s not in ("UNK","NAN")]) if "BorrowerState" in df else []
    chosen_states = st.sidebar.multiselect("State", states)
    tiers = st.sidebar.multiselect("Risk tier", ["High","Medium","Low"], default=["High","Medium","Low"])
    industries = sorted(df["industry"].dropna().unique()) if "industry" in df else []
    chosen_ind = st.sidebar.multiselect("Industry", industries)
    work = df.copy()
    if chosen_states: work = work[work["BorrowerState"].isin(chosen_states)]
    if tiers: work = work[work["risk_tier"].isin(tiers)]
    if chosen_ind: work = work[work["industry"].isin(chosen_ind)]
else:
    work = df

if df.empty:
    st.error("No demo database was found. Run scripts/build_demo_db.py or set PPP_DATA_PATH to the PPP CSV/ZIP.")
    st.stop()

mode = str(df.get("scoring_mode", pd.Series(["Unknown"])).iloc[0])
st.sidebar.markdown("---")
st.sidebar.caption(f"Scoring mode: {mode}")
st.sidebar.caption("Risk flags prioritize review; they do not establish fraud.")


def metric_card(label, value, note=""):
    st.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="subtle">{note}</div></div>', unsafe_allow_html=True)


def title(kicker, heading, subtitle):
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(heading)
    st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)


def overview():
    title("Portfolio monitoring", "Executive Overview", "Start broad, identify concentrations, then move into individual records for review.")
    hi = work[work["risk_tier"] == "High"]
    cols = st.columns(5)
    with cols[0]: metric_card("Loans in view", number(len(work)))
    with cols[1]: metric_card("Loan value", money(work["CurrentApprovalAmount"].sum()))
    with cols[2]: metric_card("High-risk queue", number(len(hi)), "Top-priority review tier")
    with cols[3]: metric_card("High-risk value", money(hi["CurrentApprovalAmount"].sum()))
    with cols[4]: metric_card("High-risk share", f"{len(hi)/max(len(work),1):.1%}")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(work, x="risk_score", nbins=35, title="Risk score distribution", labels={"risk_score":"Risk score"})
        fig.update_layout(height=330, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sample = work.sample(min(len(work), 7000), random_state=7) if len(work)>7000 else work
        fig = px.scatter(sample, x="CurrentApprovalAmount", y="risk_score", color="risk_tier", hover_name="BorrowerName",
                         hover_data=["BorrowerState","industry","JobsReported"], log_x=True,
                         category_orders={"risk_tier":["Low","Medium","High"]}, title="Loan amount vs. risk score")
        fig.update_layout(height=330, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        tmp = (work.assign(flag=work["risk_tier"].eq("High").astype(int)).groupby("industry", as_index=False)
               .agg(loans=("LoanNumber","size"), high=("flag","sum")))
        tmp["high_rate"] = tmp["high"] / tmp["loans"]
        tmp = tmp[tmp["loans"]>=20].nlargest(10,"high_rate").sort_values("high_rate")
        fig = px.bar(tmp, x="high_rate", y="industry", orientation="h", title="Highest high-risk rates by industry",
                     labels={"high_rate":"High-risk rate","industry":""})
        fig.update_xaxes(tickformat=".1%")
        fig.update_layout(height=340, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "BorrowerState" in work:
            sm = (work[~work["BorrowerState"].isin(["UNK","NAN"])].assign(flag=work.loc[~work["BorrowerState"].isin(["UNK","NAN"]),"risk_tier"].eq("High").astype(int))
                  .groupby("BorrowerState", as_index=False).agg(loans=("LoanNumber","size"), high=("flag","sum")))
            sm["high_rate"] = sm["high"] / sm["loans"]
            fig = px.choropleth(sm, locations="BorrowerState", locationmode="USA-states", color="high_rate", scope="usa",
                                title="Geographic concentration", labels={"high_rate":"High-risk rate"})
            fig.update_layout(height=340, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Highest-priority records")
    view_cols = [c for c in ["risk_score","BorrowerName","CurrentApprovalAmount","BorrowerState","industry","JobsReported","primary_risk_driver"] if c in work]
    top = work.nlargest(20,"risk_score")[view_cols].copy()
    st.dataframe(top, use_container_width=True, hide_index=True, column_config={
        "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0,max_value=1,format="%.3f"),
        "CurrentApprovalAmount": st.column_config.NumberColumn("Loan amount", format="$%,.0f")
    })


def explorer():
    title("Investigator workspace", "Anomaly Explorer", "Move from a prioritized queue to a single borrower, peer context, and explainable signals.")
    q = st.text_input("Search borrower or loan number", placeholder="e.g., construction, 9547507704")
    cur = work.copy()
    if q:
        mask = cur["BorrowerName"].fillna("").str.contains(q, case=False, regex=False) | cur["LoanNumber"].astype(str).str.contains(q, case=False, regex=False)
        cur = cur[mask]
    left, right = st.columns([1.55,1])
    with left:
        st.subheader("Prioritized review queue")
        queue = cur.nlargest(min(250,len(cur)), "risk_score").copy()
        queue["selection"] = False
        show = ["selection","risk_score","risk_tier","BorrowerName","CurrentApprovalAmount","BorrowerState","industry","primary_risk_driver"]
        edited = st.data_editor(queue[show], use_container_width=True, hide_index=False, disabled=[c for c in show if c!="selection"],
                                column_config={"selection":st.column_config.CheckboxColumn("Select"),
                                               "risk_score":st.column_config.ProgressColumn("Score",min_value=0,max_value=1,format="%.3f"),
                                               "CurrentApprovalAmount":st.column_config.NumberColumn("Loan",format="$%,.0f")}, key="queue")
        sel_idx = edited.index[edited["selection"]].tolist()
        idx = sel_idx[-1] if sel_idx else (queue.index[0] if len(queue) else None)
    with right:
        st.subheader("Selected loan")
        if idx is None:
            st.info("No record matches the current filters.")
            return
        r = df.loc[idx]
        badge = str(r.get("risk_tier","Low")).lower()
        st.markdown(f"### {r.get('BorrowerName','Unknown borrower')}")
        st.markdown(f'<span class="badge-{badge}">{str(r.get("risk_tier","Low")).upper()} PRIORITY</span>', unsafe_allow_html=True)
        a,b = st.columns(2)
        a.metric("Risk score", f"{r.get('risk_score',np.nan):.3f}")
        b.metric("Loan amount", money(r.get("CurrentApprovalAmount",np.nan)))
        a.metric("Jobs reported", f"{r.get('JobsReported',np.nan):,.0f}" if pd.notna(r.get('JobsReported')) else "—")
        b.metric("Loan / job", money(r.get("amount_per_job",np.nan)))
        st.markdown("#### Why this record stands out")
        for reason in [r.get("primary_risk_driver",""), r.get("secondary_risk_driver","")]:
            if reason: st.markdown(f'<div class="risk-reason">{reason}</div>', unsafe_allow_html=True)
        st.markdown("#### Peer context")
        peer = float(r.get("peer_median_loan",np.nan))
        loan = float(r.get("CurrentApprovalAmount",np.nan))
        comp = pd.DataFrame({"Measure":["Selected loan","Industry median"],"Amount":[loan,peer]})
        fig = px.bar(comp, x="Amount", y="Measure", orientation="h")
        fig.update_layout(height=220, margin=dict(l=10,r=10,t=15,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        if pd.notna(r.get("industry_loan_percentile")):
            st.caption(f"Loan amount percentile within broad industry: {r['industry_loan_percentile']:.1%}")

    st.download_button("Download current filtered queue", cur.to_csv(index=False).encode("utf-8"), "ppp_filtered_review_queue.csv", "text/csv")


def insights():
    title("Model transparency", "Model Insights", "Show how prioritization behaves, where signals come from, and what the customer should—and should not—infer.")
    c = st.columns(4)
    with c[0]: metric_card("Scoring mode", mode)
    with c[1]: metric_card("High tier", "Top 1%", "Capacity-oriented review cutoff")
    with c[2]: metric_card("Medium tier", "Next 4%")
    with c[3]: metric_card("IF signal", "Top 1%", "Structurally unusual tag")
    st.info("A high score prioritizes a record for review. It is not, by itself, a determination of fraud.")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(work, x="risk_score", color="risk_tier", nbins=35, title="Score distribution by tier")
        fig.update_layout(height=330, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        metrics=[]
        for feat,label in [("CurrentApprovalAmount","Average loan"),("JobsReported","Average jobs"),("amount_per_job","Loan / job")]:
            if feat in work:
                lo=work.loc[work.risk_tier=="Low",feat].median(); hi=work.loc[work.risk_tier=="High",feat].median()
                metrics += [{"Feature":label,"Tier":"Low","Value":lo},{"Feature":label,"Tier":"High","Value":hi}]
        mm=pd.DataFrame(metrics)
        fig=px.bar(mm, x="Feature", y="Value", color="Tier", barmode="group", title="High-tier vs. low-tier medians")
        fig.update_layout(height=330, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        drivers=work.loc[work.risk_tier.isin(["High","Medium"]),"primary_risk_driver"].value_counts().head(8).sort_values()
        fig=px.bar(x=drivers.values,y=drivers.index,orientation="h",title="Most common review signals",labels={"x":"Records","y":""})
        fig.update_layout(height=330, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        mix=work["risk_tier"].value_counts().reindex(["High","Medium","Low"]).fillna(0).reset_index()
        mix.columns=["Risk tier","Loans"]
        fig=px.pie(mix,names="Risk tier",values="Loans",hole=.55,title="Review tier mix")
        fig.update_layout(height=330, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Integration path")
    st.markdown("**PPP source → feature engineering / scoring → scored loan table → Streamlit.** The UI reads a stable scored-table contract, so the scoring backend can move from the demo Isolation Forest to the notebook's hybrid XGBoost + Isolation Forest output without redesigning the dashboard.")

if page == "Executive Overview": overview()
elif page == "Anomaly Explorer": explorer()
else: insights()
