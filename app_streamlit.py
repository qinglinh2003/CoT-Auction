import os, glob, json
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CoT Auction Demo", layout="wide")

def latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

@st.cache_data(show_spinner=False)
def load_records(csv_path):
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns:
        df["run_id"] = 1
    for col in ["k","correct","token_input","token_output","run_id"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["cost_usd","latency_s","agreement"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["acc"] = df["correct"].astype(float)
    return df

@st.cache_data(show_spinner=False)
def load_details(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for d in data:
        key = (int(d.get("run_id",1)), d["qid"], int(d["k"]))
        idx[key] = d
    return data, idx

def aggregate_by_k(df, category=None):
    view = df if category in (None, "All") else df[df["category"]==category]
    stats = view.groupby("k").agg(
        acc_mean=("acc","mean"),
        acc_std =("acc",lambda x: x.std(ddof=1) if len(x)>1 else 0.0),
        cost_mean=("cost_usd","mean"),
        cost_std =("cost_usd",lambda x: x.std(ddof=1) if len(x)>1 else 0.0),
    ).reset_index().sort_values("cost_mean")
    return stats

def pareto_front(points):
    front = []
    best = -1.0
    for _, p in points.sort_values("cost_mean").iterrows():
        if p["acc_mean"] > best:
            front.append(p)
            best = p["acc_mean"]
    return pd.DataFrame(front) if front else points.iloc[0:0]

def reco_iso_accuracy(stats_df, target):
    feas = stats_df[stats_df["acc_mean"]>=target]
    if len(feas):
        row = feas.sort_values(["cost_mean","k"]).iloc[0]
        return dict(feasible=True, k=int(row.k), acc=row.acc_mean, cost=row.cost_mean)
    row = stats_df.sort_values(["acc_mean","k"], ascending=[False,True]).iloc[0]
    return dict(feasible=False, k=int(row.k), acc=row.acc_mean, cost=row.cost_mean)

def reco_iso_cost(stats_df, budget):
    feas = stats_df[stats_df["cost_mean"]<=budget]
    if len(feas):
        row = feas.sort_values(["acc_mean","k"], ascending=[False,True]).iloc[0]
        return dict(feasible=True, k=int(row.k), acc=row.acc_mean, cost=row.cost_mean)
    row = stats_df.sort_values(["cost_mean","k"]).iloc[0]
    return dict(feasible=False, k=int(row.k), acc=row.acc_mean, cost=row.cost_mean)

st.title("CoT Auction (Demo)")
st.caption("Tune the reasoning budget and see the accuracy–cost trade-off. Pareto frontier, iso-accuracy/iso-cost recommendations, and drill-down chains.")

default_records = latest_file("outputs/records_*.csv")
default_details = latest_file("outputs/details_*.json")
default_report  = latest_file("outputs/report_*.json")

with st.sidebar:
    st.subheader("Data sources")
    rec_path = st.text_input("Records CSV path", value=default_records or "")
    det_path = st.text_input("Details JSON path", value=default_details or "")
    rep_path = st.text_input("Report JSON path",  value=default_report or "")

    if not rec_path or not os.path.exists(rec_path):
        st.error("Records CSV not found. Run the evaluator to generate outputs/records_*.csv.")
        st.stop()
    if not det_path or not os.path.exists(det_path):
        st.warning("Details JSON not found. Drill-down panel will be disabled.")

    df = load_records(rec_path)

    cats = ["All"] + sorted(df["category"].unique().tolist())
    sel_cat = st.selectbox("Category", cats, index=0)

    st.markdown("---")
    st.subheader("Recommendation targets")
    col_a, col_b = st.columns(2)
    with col_a:
        iso_acc = st.number_input("Target accuracy (iso-accuracy)", min_value=0.5, max_value=1.0, value=0.90, step=0.01)
    with col_b:
        iso_cost = st.number_input("Budget (iso-cost, $)", min_value=0.0, value=float(df["cost_usd"].mean() or 0.0), step=0.01, format="%.2f")

    show_dominated = st.checkbox("Dim dominated points", value=True)
    st.caption("Dominated = worse accuracy at same cost or higher cost at same accuracy.")

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Pareto chart (Accuracy vs Cost)")
    stats = aggregate_by_k(df, category=sel_cat)
    if stats.empty:
        st.info("No data to display.")
    else:
        front = pareto_front(stats)
        front_k = set(front["k"].tolist())

        fig, ax = plt.subplots()
        for _, r in stats.iterrows():
            kwargs = dict(fmt='o', capsize=3)
            if show_dominated and (int(r.k) not in front_k):
                kwargs.update(dict(alpha=0.35))
            ax.errorbar(r.cost_mean, r.acc_mean, xerr=r.cost_std, yerr=r.acc_std, **kwargs)
            ax.annotate(f'k={int(r.k)}', (r.cost_mean, r.acc_mean), xytext=(3,3), textcoords="offset points",
                        fontsize=9, alpha=(0.5 if (show_dominated and int(r.k) not in front_k) else 1.0))
        if len(front) >= 2:
            ax.plot(front["cost_mean"], front["acc_mean"], linewidth=2)
        ax.set_xlabel("Average cost (USD)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Recommendations by category")
    rec_rows = []
    for cat in sorted(df["category"].unique().tolist()):
        s = aggregate_by_k(df, category=cat)
        if s.empty:
            continue
        ra = reco_iso_accuracy(s, iso_acc)
        rc = reco_iso_cost(s, iso_cost)
        rec_rows.append(dict(
            category=cat,
            iso_acc_target=iso_acc, k_for_iso_acc=ra["k"], acc=round(ra["acc"],3), cost=round(ra["cost"],4), feasible=ra["feasible"],
            iso_cost_budget=iso_cost, k_for_iso_cost=rc["k"], acc2=round(rc["acc"],3), cost2=round(rc["cost"],4), feasible2=rc["feasible"]
        ))
    if rec_rows:
        st.dataframe(pd.DataFrame(rec_rows))

with right:
    st.subheader("Drill-down (multiple chains and voting)")
    if det_path and os.path.exists(det_path):
        details, idx = load_details(det_path)
        runs = sorted({ int(d.get("run_id",1)) for d in details })
        sel_run = st.selectbox("Run", runs, index=len(runs)-1)
        qids = sorted({ d["qid"] for d in details if d.get("category")==sel_cat } if sel_cat!="All" 
                      else { d["qid"] for d in details })
        if not qids:
            st.info("No details for this category.")
        else:
            sel_qid = st.selectbox("Question ID", qids)
            ks = sorted({ int(d["k"]) for d in details if d["qid"]==sel_qid })
            sel_k = st.selectbox("k (reasoning budget)", ks)
            key = (int(sel_run), sel_qid, int(sel_k))
            d = idx.get(key)
            if not d:
                st.warning("This (run, qid, k) was not found in details. Try another run or k.")
            else:
                st.caption(d.get("prompt",""))
                vh = d.get("vote_hist", {})
                if vh:
                    fig2, ax2 = plt.subplots()
                    items = sorted(vh.items(), key=lambda x: (-x[1], x[0]))
                    ax2.bar([k for k,v in items], [v for k,v in items])
                    ax2.set_title(f"Voting distribution (agreement={d.get('agreement',0.0):.2f})")
                    st.pyplot(fig2, clear_figure=True)
                st.markdown(f"**Majority vote**: `{d.get('vote','')}` · **Correct**: {d.get('correct',0)}")
                for i, c in enumerate(d.get("chains", []), 1):
                    with st.expander(f"Chain #{i} · answer `{c.get('answer','')}` · latency={c.get('latency_s',0.0)}s · usage={c.get('usage',{})}"):
                        st.write(c.get("brief","(no rationale)"))
    else:
        st.info("Details JSON not provided, drill-down panel disabled.")

st.markdown("---")
st.caption("Note: Cost is estimated from per-1K-token prices. With local Ollama configs it is typically 0; adjust config to visualize monetary trade-offs.")
