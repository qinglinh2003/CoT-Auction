cat > app_streamlit.py << 'PY'
import os, glob, json, io
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

def aggregate_by_k(df, category=None, x_col="cost_usd"):
    view = df if category in (None, "All") else df[df["category"]==category]
    stats = view.groupby("k").agg(
        acc_mean=("acc","mean"),
        acc_std =("acc",lambda x: x.std(ddof=1) if len(x)>1 else 0.0),
        x_mean =(x_col,"mean"),
        x_std  =(x_col,lambda x: x.std(ddof=1) if len(x)>1 else 0.0),
    ).reset_index().sort_values("x_mean")
    return stats

def pareto_front(points_df):
    front = []
    best = -1.0
    for _, p in points_df.sort_values("x_mean").iterrows():
        if p["acc_mean"] > best:
            front.append(p)
            best = p["acc_mean"]
    return pd.DataFrame(front) if front else points_df.iloc[0:0]

def reco_iso_accuracy(stats_df, target):
    feas = stats_df[stats_df["acc_mean"]>=target]
    if len(feas):
        row = feas.sort_values(["x_mean","k"]).iloc[0]
        return dict(feasible=True, k=int(row.k), acc=row.acc_mean, x=row.x_mean)
    row = stats_df.sort_values(["acc_mean","k"], ascending=[False,True]).iloc[0]
    return dict(feasible=False, k=int(row.k), acc=row.acc_mean, x=row.x_mean)

def reco_iso_x(stats_df, budget):
    feas = stats_df[stats_df["x_mean"]<=budget]
    if len(feas):
        row = feas.sort_values(["acc_mean","k"], ascending=[False,True]).iloc[0]
        return dict(feasible=True, k=int(row.k), acc=row.acc_mean, x=row.x_mean)
    row = stats_df.sort_values(["x_mean","k"]).iloc[0]
    return dict(feasible=False, k=int(row.k), acc=row.acc_mean, x=row.x_mean)

def get_qp():
    try:
        return st.experimental_get_query_params()
    except:
        return {}

def set_qp(d):
    try:
        st.experimental_set_query_params(**d)
    except:
        pass

st.title("CoT Auction (Demo)")
st.caption("Tune the reasoning budget and see the accuracy–cost/latency trade-off. Pareto frontier, iso-target recommendations, and drill-down chains.")

default_records = latest_file("outputs/records_*.csv")
default_details = latest_file("outputs/details_*.json")
default_report  = latest_file("outputs/report_*.json")
qp = get_qp()

with st.sidebar:
    st.subheader("Data sources")
    rec_path = st.text_input("Records CSV path", value=qp.get("rec", [default_records or ""])[0])
    det_path = st.text_input("Details JSON path", value=qp.get("det", [default_details or ""])[0])
    rep_path = st.text_input("Report JSON path",  value=qp.get("rep", [default_report or ""])[0])

    if not rec_path or not os.path.exists(rec_path):
        st.error("Records CSV not found. Run the evaluator to generate outputs/records_*.csv.")
        st.stop()
    if not det_path or not os.path.exists(det_path):
        st.warning("Details JSON not found. Drill-down panel will be disabled.")

    df = load_records(rec_path)

    cats = ["All"] + sorted(df["category"].unique().tolist())
    cat_default = qp.get("cat", ["All"])[0]
    cat_default = cat_default if cat_default in cats else "All"
    sel_cat = st.selectbox("Category", cats, index=cats.index(cat_default))

    x_mode_default = qp.get("x", ["cost"])[0]
    x_mode = st.radio("X-axis metric", ["Cost (USD)", "Latency (s)"], horizontal=True,
                      index=(0 if x_mode_default=="cost" else 1))
    x_col = "cost_usd" if x_mode.startswith("Cost") else "latency_s"
    x_label = "Average cost (USD)" if x_col=="cost_usd" else "Average latency (s)"

    st.markdown("---")
    st.subheader("Recommendation targets")
    col_a, col_b = st.columns(2)
    with col_a:
        try:
            acc_default = float(qp.get("acc", [0.90])[0])
        except:
            acc_default = 0.90
        iso_acc = st.number_input("Target accuracy (iso-accuracy)", min_value=0.5, max_value=1.0, value=acc_default, step=0.01)
    with col_b:
        default_budget = float(df[x_col].mean() or 0.0)
        try:
            budget_default = float(qp.get("budget", [default_budget])[0])
        except:
            budget_default = default_budget
        iso_x = st.number_input(f"Budget (iso-{ 'cost' if x_col=='cost_usd' else 'latency' })", min_value=0.0, value=budget_default, step=0.01, format="%.2f")

    dom_default = qp.get("dom", ["1"])[0] == "1"
    show_dominated = st.checkbox("Dim dominated points", value=dom_default)

    st.markdown("---")
    if st.button("Share this view (update URL)"):
        set_qp({
            "rec": [rec_path],
            "det": [det_path],
            "rep": [rep_path],
            "cat": [sel_cat],
            "x": ["cost" if x_col=="cost_usd" else "lat"],
            "acc": [f"{iso_acc:.2f}"],
            "budget": [f"{iso_x:.4f}"],
            "dom": ["1" if show_dominated else "0"],
        })
        st.success("URL updated. Copy it from the browser address bar.")

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader(f"Pareto chart (Accuracy vs { 'Cost' if x_col=='cost_usd' else 'Latency' })")
    stats = aggregate_by_k(df, category=sel_cat, x_col=x_col)
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
            ax.errorbar(r.x_mean, r.acc_mean, xerr=r.x_std, yerr=r.acc_std, **kwargs)
            ax.annotate(f'k={int(r.k)}', (r.x_mean, r.acc_mean), xytext=(3,3), textcoords="offset points",
                        fontsize=9, alpha=(0.5 if (show_dominated and int(r.k) not in front_k) else 1.0))
        if len(front) >= 2:
            ax.plot(front["x_mean"], front["acc_mean"], linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, clear_figure=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        st.download_button("Download chart PNG", data=buf.getvalue(), file_name="pareto_chart.png", mime="image/png")

    st.subheader("Recommendations by category")
    rec_rows = []
    for cat in sorted(df["category"].unique().tolist()):
        s = aggregate_by_k(df, category=cat, x_col=x_col)
        if s.empty:
            continue
        ra = reco_iso_accuracy(s, iso_acc)
        rx = reco_iso_x(s, iso_x)
        rec_rows.append(dict(
            category=cat,
            iso_accuracy_target=iso_acc, k_for_iso_accuracy=ra["k"], acc=round(ra["acc"],3), x=round(ra["x"],4), feasible=ra["feasible"],
            iso_budget=iso_x, k_for_iso_budget=rx["k"], acc2=round(rx["acc"],3), x2=round(rx["x"],4), feasible2=rx["feasible"]
        ))
    if rec_rows:
        rec_df = pd.DataFrame(rec_rows)
        st.dataframe(rec_df)
        md_lines = ["# CoT Auction Recommendations\n"]
        for r in rec_rows:
            md_lines.append(f"- [{r['category']}] iso-acc {r['iso_accuracy_target']:.2f}: k={r['k_for_iso_accuracy']} (acc={r['acc']:.3f}, x={r['x']:.4f}, feasible={r['feasible']})")
            md_lines.append(f"  [{r['category']}] iso-budget {r['iso_budget']:.2f}: k={r['k_for_iso_budget']} (acc={r['acc2']:.3f}, x={r['x2']:.4f}, feasible={r['feasible2']})")
        md_blob = "\n".join(md_lines).encode("utf-8")
        st.download_button("Download recommendations (Markdown)", data=md_blob, file_name="recommendations.md", mime="text/markdown")

    st.subheader("Data exports")
    if os.path.exists(rec_path):
        st.download_button("Download records CSV", data=open(rec_path, "rb").read(), file_name=os.path.basename(rec_path), mime="text/csv")
    if det_path and os.path.exists(det_path):
        st.download_button("Download details JSON", data=open(det_path, "rb").read(), file_name=os.path.basename(det_path), mime="application/json")

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
st.caption("Note: If cost is zero with local models, switch X-axis to latency to compare speed vs accuracy. Use the sidebar button to update the URL as a shareable permalink.")
PY
