import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from autorank import autorank, plot_stats
import matplotlib.pyplot as plt

import data.utils as data_utils
import constant
st.set_page_config(page_title="ATLASExplorer")


with st.sidebar:
    st.markdown('# ATLASExplorer Engine')
    st.markdown('## Please Select :point_down:')

    metric_name = st.selectbox(
        "Evaluation Measure",
        constant.list_measures,
        key="metric_name"
    )

    # Clear the "other" dataset widget state when switching measures
    if metric_name == "WCSR (Critical Diagram)":
        st.session_state.pop("dataset_single", None)
    else:
        st.session_state.pop("dataset_multi", None)

    # Dataset selection:
    # - ONLY Critical Diagram => multiple datasets
    # - Everything else => exactly one dataset
    if metric_name == "WCSR (Critical Diagram)":
        # all_dataset = st.checkbox("Select all", key="all_dataset_multi")

        datasets = st.multiselect(
            "Select Datasets",
            options=constant.datasets,
            default=constant.datasets if all_dataset else [],
            key="dataset_multi"
        )
    else:
        ds = st.selectbox(
            "Select Dataset",
            options=constant.datasets,
            key="dataset_single"
        )
        datasets = [ds]  # always a list of length 1

    # Algorithms (keep as-is; you can add the same idea if needed)
    algorithms = st.multiselect(
        "Select Algorithms",
        constant.models,
        key="algorithms_multi"
    )
    

tab_desc, tab_benchmark, tab_eva = st.tabs(["Overview", "Benchmark", "Evaluation"]) 

def visualize_graph(dataset,algorithms):
    df=data_utils.get_data_for_speedup_recall_graph(dataset,algorithms)
    # fig = px.line(df, x="recall", y="qps", color="model")
    # fig.update_yaxes(type="log")
    # recall_threshold = 0.85

    # fig.add_vline(
    #     x=recall_threshold,
    #     line_width=2,
    #     line_dash="dash",
    #     line_color="black",
    #     annotation_text="Recall threshold",
    #     annotation_position="top left",
    #     editable=True
    # )

    # st.plotly_chart(fig, use_container_width=True, config={"editable": True})
    recall_thr = st.slider("Recall threshold", 0.0, 1.0, 0.85, 0.01)

    # --- Build a consistent color map for models (shared across charts) ---
    models = sorted(df["model"].unique())
    palette = px.colors.qualitative.Plotly
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    # =========================
    # 1) Recall–QPS line chart
    # =========================
    df_line = df.sort_values(["model", "recall"]).copy()

    fig_curve = px.line(
        df_line,
        x="recall",
        y="qps",
        color="model",
        color_discrete_map=color_map,
        title="Evaluation measure: Recall–QPS trade off",
    )

    # Log scale on QPS (optional; remove if you want linear)
    fig_curve.update_yaxes(type="log")

    # Vertical recall threshold line
    fig_curve.add_vline(
        x=recall_thr,
        line_width=2,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Recall ≥ {recall_thr:.2f}",
        annotation_position="top left",
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    # ==============================================
    # 2) Best QPS per model given recall threshold
    #    (max qps among points with recall >= thr)
    # ==============================================
    df_valid = df[df["recall"] >= recall_thr].copy()

    if df_valid.empty:
        st.warning("No points meet the selected recall threshold.")
    else:
        best = (
            df_valid.groupby("model", as_index=False)
            .agg(best_qps=("qps", "max"), best_recall=("recall", "max"))
            .sort_values("best_qps", ascending=False)
        )

        fig_bar = px.bar(
            best,
            x="best_qps",
            y="model",
            orientation="h",
            text="best_qps",
            color="model",                 # color bars by model
            color_discrete_map=color_map,  # SAME colors as line chart
            title=f"Best QPS at Recall ≥ {recall_thr:.2f}",
        )

        fig_bar.update_traces(
            texttemplate="%{text:.0f} qps",
            textposition="outside",
            cliponaxis=False
        )

        # Put highest at top (like your screenshot)
        fig_bar.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False
        )

        st.plotly_chart(fig_bar, use_container_width=True)

def visualize_critical_diagram(dataset,algorithms):
    if len(dataset)>1:
        auc,critical_datasets=data_utils.get_data_for_critical_diagram(dataset,algorithms)
        print(auc,critical_datasets)
        df = pd.DataFrame(auc, index=critical_datasets)
        result = autorank(df, alpha=0.05, verbose=False, force_mode="nonparametric")
        fig = plt.figure(figsize=(8, 2))
        ax = plot_stats(result, ax=fig.gca(),allow_insignificant=True)
        st.pyplot(fig, clear_figure=False)
    elif len(dataset) == 1 and len(algorithms) >= 2:
        wcsr_data, _ = data_utils.get_data_for_critical_diagram(dataset, algorithms)
        print(wcsr_data)

        df_wcsr = pd.DataFrame(
            [{"model": m, "wcsr": float(vals[0])}
            for m, vals in wcsr_data.items()
            if vals is not None and len(vals) > 0]
        )

        if df_wcsr.empty:
            st.warning("No WCSR values available to plot.")
        else:
            # Sort so it reads nicely (optional)
            df_wcsr = df_wcsr.sort_values("wcsr", ascending=False)

            # 1-column "heatmap": rows=models, col=WCSR
            heat_df = df_wcsr.set_index("model")[["wcsr"]]

            import plotly.express as px

            fig = px.imshow(
                heat_df,
                text_auto=".1f",          # show values in cells
                aspect="auto",
                labels=dict(x="", y="Algorithm", color="WCSR"),
            )

            fig.update_layout(
                title=f"WCSR Heatmap (Dataset: {dataset[0]})",
                xaxis_title="",
                yaxis_title="Algorithm",
                margin=dict(l=10, r=10, t=50, b=10),
            )

            # Put the single column label on top (looks nicer)
            fig.update_xaxes(side="top")

            st.plotly_chart(fig, use_container_width=True)
        

with tab_desc:
    st.markdown("## 🏄 Dive into ATLASExplorer")
    image = Image.open('figures/final_paper-engine-overview.png')
    st.image(image)
    st.markdown(constant.description_intro)
    st.markdown("#### User Manual")
    image = Image.open('figures/final_paper-demo_functions.png')
    st.image(image, caption='The main frames of ATLASExplorer Engine')
    st.markdown(constant.User_Manual)
    st.markdown(constant.Contributors)
    
with tab_benchmark:
    st.markdown('#### Taxonomy of Automated Solutions for TSAD')
    image = Image.open('figures/final_paper-taxonomy_newcolor.png')
    st.image(image)

with tab_eva:
    st.markdown('#### Evaluation measure: {}'.format(metric_name))
    if metric_name=='Recall-QPS trade off':
        if len(datasets) == 0:
            st.markdown("#### :heavy_exclamation_mark: Note: Please select a dataset in the left :point_left: panel")
        if len(datasets) >1:
            st.markdown("#### :heavy_exclamation_mark: Note: You are selecting more than 1 dataset")
        if len(algorithms) == 0:
            st.markdown("#### :heavy_exclamation_mark: Note: Please select algorithms in the left :point_left: panel")
        #Show the plot when dataset and algorithm is selected
        if len(datasets)==1 and len(algorithms)>0:
            visualize_graph(datasets[0],algorithms)
    if metric_name=='WCSR (Critical Diagram)':
        if len(algorithms) <3:
            st.markdown("#### :heavy_exclamation_mark: Note: You need to select at least 3 methods")
        elif len(datasets) <= len(algorithms):
            st.markdown("#### :heavy_exclamation_mark: Note: You need to select more datasets than methods")
        else:
            visualize_critical_diagram(datasets,algorithms)
    
    if metric_name=='WCSR (Per Dataset Comparison)':
        if len(algorithms) <2:
            st.markdown("#### :heavy_exclamation_mark: Note: You need to select at least 2 methods")
        elif len(datasets) != 1:
            st.markdown("#### :heavy_exclamation_mark: Note: You need to select exactly one dataset for the analysis")
        else:
            visualize_critical_diagram(datasets,algorithms)
            
        
            
