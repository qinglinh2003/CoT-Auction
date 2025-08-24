import os, glob, json, io, yaml
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

def list_output_records():
    files = glob.glob("outputs/records_*.csv")
    files.sort(key=os.path.getmtime, reverse=True)
    return files

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

def infer_strategy_from_cfg(cfg_path):
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        ks = y.get("strategy", {}).get("k_list", None)
        ms = y.get("strategy", {}).get("max_steps", None)
        tp = y.get("strategy", {}).get("temperature", None)
        return dict(k_list=ks, max_steps=ms, temperature=tp)
    except:
        return None

def infer_strategy_from_records(df):
    ks = sorted(df["k"].dropna().astype(int).unique().tolist())
    return dict(k_list=ks, max_steps=None, temperature=None)

def derive_sibling(path, kind):
    base = os.path.basename(path)
    if base.startswith("records_") and base.endswith(".csv"):
        ts = base[len("records_"):-len(".csv")]
        if kind == "details":
            cand = os.path.join(os.path.dirname(path), f"details_{ts}.json")
            return cand if os.path.exists(cand) else latest_file("outputs/details_*.json")
        if kind == "report":
            cand = os.path.join(os.path.dirname(path), f"report_{ts}.json")
            return cand if os.path.exists(cand) else latest_file("outputs/report_*.json")
    return None

st.title("CoT Auction (Demo)")
st.caption("Tune the reasoning budget and see the accuracy–cost/latency trade-off. Pareto frontier, iso-target recommendations, and drill-down chains.")

records_candidates = list_output_records()
default_records = records_candidates[0] if records_candidates else latest_file("outputs/records_*.csv")
with st.sidebar:
    st.subheader("Data sources")
    rec_sel = st.selectbox("Select records run", records_candidates, index=0 if records_candidates else None, placeholder="No records found")
    rec_path = st.text_input("Records CSV path", value=rec_sel or (default_records or ""))

    det_guess = derive_sibling(rec_path, "details") or ""
    rep_guess = derive_sibling(rec_path, "report") or ""
    det_path = st.text_input("Details JSON path", value=det_guess)
    rep_path = st.text_input("Report JSON path",  value=rep_guess)

    cfg_default = "configs/ollama.yaml" if os.path.exists("configs/ollama.yaml") else ""
    cfg_path = st.text_input("Config YAML path (optional)", value=cfg_default)

    if not rec_path or not os.path.exists(rec_path):
        st.error("Records CSV not found. Run the evaluator to generate outputs/records_*.csv.")
        st.stop()
    if det_path and not os.path.exists(det_path):
        st.warning("Details JSON not found. Drill-down panel will be disabled.")

    df = load_records(rec_path)

    cats = ["All"] + sorted(df["category"].unique().tolist())
    sel_cat = st.selectbox("Category", cats, index=0)

    x_mode = st.radio("X-axis metric", ["Cost (USD)", "Latency (s)"], horizontal=True)
    x_col = "cost_usd" if x_mode.startswith("Cost") else "latency_s"
    x_label = "Average cost (USD)" if x_col=="cost_usd" else "Average latency (s)"

    st.markdown("---")
    st.subheader("Clarity options")
    col1, col2 = st.columns(2)
    with col1:
        show_errorbars = st.checkbox("Show error bars", value=False)
        show_labels = st.checkbox("Show k labels", value=True)
        jitter_points = st.checkbox("Offset overlapping points", value=True)
        connect_sequence = st.checkbox("Connect ks in order", value=True)
    with col2:
        frontier_only = st.checkbox("Show frontier only", value=False)
        auto_zoom_y = st.checkbox("Auto-zoom Y to data", value=True)
        dim_dominated = st.checkbox("Dim dominated points", value=True)

    st.markdown("---")
    st.subheader("k filter")
    all_k = sorted(df["k"].astype(int).unique().tolist())
    k_filter = st.multiselect("Show ks", all_k, default=all_k)

    st.markdown("---")
    st.subheader("Recommendation targets")
    col_a, col_b = st.columns(2)
    with col_a:
        iso_acc = st.number_input("Target accuracy (iso-accuracy)", min_value=0.5, max_value=1.0, value=0.90, step=0.01)
    with col_b:
        default_budget = float(df[x_col].mean() or 0.0)
        iso_x = st.number_input(f"Budget (iso-{ 'cost' if x_col=='cost_usd' else 'latency' })", min_value=0.0, value=default_budget, step=0.01, format="%.2f")

    st.markdown("---")
    st.subheader("Current strategy (read-only)")
    strat_cfg = infer_strategy_from_cfg(cfg_path) if cfg_path else None
    strat_rec = infer_strategy_from_records(df)
    st.write({
        "k_list": strat_cfg["k_list"] if strat_cfg and strat_cfg.get("k_list") else strat_rec["k_list"],
        "max_steps": strat_cfg.get("max_steps") if strat_cfg else None,
        "temperature": strat_cfg.get("temperature") if strat_cfg else None
    })

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader(f"Pareto chart (Accuracy vs { 'Cost' if x_col=='cost_usd' else 'Latency' })")
    stats = aggregate_by_k(df, category=sel_cat, x_col=x_col)
    if k_filter:
        stats = stats[stats["k"].isin(k_filter)]
    if stats.empty:
        st.info("No data to display.")
    else:
        front = pareto_front(stats)
        front_k = set(front["k"].tolist())
        pts = stats[stats["k"].isin(front_k)] if frontier_only else stats

        # deterministic small offset to avoid overlap
        x_vals = pts["x_mean"].to_numpy()
        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        x_range = max(1e-6, x_max - x_min)
        jitter = 0.02 * x_range if jitter_points else 0.0

        # shape/color mapping by k
        shapes = ['o','s','^','D','P','X','v','>','<','h']
        k_sorted = sorted(pts["k"].astype(int).unique().tolist())
        shape_map = {k: shapes[i % len(shapes)] for i, k in enumerate(k_sorted)}
        colors = plt.cm.tab10(np.linspace(0, 1, len(k_sorted)))
        color_map = {k: colors[i] for i, k in enumerate(k_sorted)}

        fig, ax = plt.subplots()

        for _, row in pts.iterrows():
            k = int(row.k)
            base_x = row.x_mean
            offset_sign = (k_sorted.index(k) - (len(k_sorted)-1)/2.0)
            x_plot = base_x + offset_sign * jitter
            alpha = 1.0
            if dim_dominated and (k not in front_k):
                alpha = 0.35
            if show_errorbars:
                ax.errorbar(x_plot, row.acc_mean,
                            xerr=row.x_std, yerr=row.acc_std,
                            fmt=shape_map[k], capsize=3, alpha=alpha, color=color_map[k])
            else:
                ax.plot(x_plot, row.acc_mean, shape_map[k], alpha=alpha, color=color_map[k])
            if show_labels:
                ax.annotate(f'k={k}', (x_plot, row.acc_mean),
                            xytext=(3,3), textcoords="offset points",
                            fontsize=9, alpha=alpha)

        if connect_sequence:
            seq = pts.sort_values("k")
            ax.plot(seq["x_mean"].to_numpy(), seq["acc_mean"].to_numpy(),
                    linewidth=2, color="#6f61ff", alpha=0.6)

        if len(front) >= 2 and not connect_sequence:
            ax.plot(front["x_mean"], front["acc_mean"], linewidth=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy")
        if auto_zoom_y:
            yvals = pts["acc_mean"].to_numpy()
            low, high = float(np.min(yvals)), float(np.max(yvals))
            pad = max(0.01, (high - low) * 0.15)
            ax.set_ylim(max(0.0, low - pad), min(1.0, high + pad))
        else:
            ax.set_ylim(0,1.0)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, clear_figure=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        st.download_button("Download chart PNG", data=buf.getvalue(), file_name="pareto_chart.png", mime="image/png")

    st.subheader("Recommendations by category")
    rec_rows = []
    for cat in sorted(df["category"].unique().tolist()):
        s = aggregate_by_k(df, category=cat, x_col=x_col)
        if k_filter:
            s = s[s["k"].isin(k_filter)]
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
        st.dataframe(pd.DataFrame(rec_rows))

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
        # filter qids by category if chosen
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
                    agr = float(d.get('agreement', 0.0))
                    ax2.set_title(f"Voting distribution (agreement={agr:.2f})")
                    st.pyplot(fig2, clear_figure=True)
                st.markdown(f"**Majority vote**: `{d.get('vote','')}` · **Correct**: {d.get('correct',0)}")
                for i, c in enumerate(d.get("chains", []), 1):
                    with st.expander(f"Chain #{i} · answer `{c.get('answer','')}` · latency={c.get('latency_s',0.0)}s · usage={c.get('usage',{})}"):
                        st.write(c.get("brief","(no rationale)"))
    else:
        st.info("Details JSON not provided, drill-down panel disabled.")

st.markdown("---")
st.caption("Tip: Use 'Frontier only' or 'Connect ks in order', hide error bars, offset points, and auto-zoom for the cleanest view.")
