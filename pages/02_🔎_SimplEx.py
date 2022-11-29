# Imports
# Standard
import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import dill as pkl
import sklearn

# Interpretability
from interpretability.interpretability_models import simplex_explainer
from interpretability.interpretability_models.utils import data, io

# Page Config
st.set_page_config(
    page_title="Interpretability Suite",
    page_icon="images/ccaim_logo-no_text.png",
    layout="wide",
)

# Demo util functions
def write_time_series_explaination_to_app(my_explainer, time_steps_to_display=7):
    def highlight(x):
        return pd.DataFrame(
            importance_df_colors.values, index=x.index, columns=x.columns
        )

    example_importance_threshold = 0.05

    test_record_last_time_step = (
        my_explainer.explanation.test_record[
            ~np.all(my_explainer.explanation.test_record == 0, axis=1)
        ].shape[0]
        - 1
    )
    test_record_df = (
        pd.DataFrame(
            my_explainer.explanation.test_record[
                test_record_last_time_step
                - (time_steps_to_display - 1) : test_record_last_time_step
                + 1,
            ],
            columns=my_explainer.feature_names,
            index=[
                f"(t_max) - {i}" if i != 0 else "(t_max)"
                for i in reversed(range(time_steps_to_display))
            ],
        )
        if time_steps_to_display <= test_record_last_time_step
        else pd.DataFrame(
            my_explainer.explanation.test_record[0 : test_record_last_time_step + 1],
            columns=my_explainer.feature_names,
            index=[
                f"(t_max) - {i}" if i != 0 else "(t_max)"
                for i in reversed(range(test_record_last_time_step + 1))
            ],
        )
    )
    # Write test record to app
    st.write("### Test Record")
    st.write(test_record_df.transpose())
    st.write("### Corpus")

    # Corpus Feature values
    last_time_step_idx = [
        my_explainer.explanation.corpus_breakdown[j][
            ~np.all(my_explainer.explanation.corpus_breakdown[j] == 0, axis=1)
        ].shape[0]
        - 1
        for j in range(len(my_explainer.explanation.corpus_breakdown))
    ]

    corpus_dfs = [
        pd.DataFrame(
            my_explainer.explanation.corpus_breakdown[j][
                idx - (time_steps_to_display - 1) : idx + 1
            ],
            index=[
                f"(t_max) - {i}" if i != 0 else "(t_max)"
                for i in reversed(range(time_steps_to_display))
            ],
            columns=my_explainer.feature_names,
        )
        if time_steps_to_display <= idx
        else pd.DataFrame(
            my_explainer.explanation.corpus_breakdown[j][0 : idx + 1],
            index=[
                f"(t_max) - {i}" if i != 0 else "(t_max)" for i in reversed(range(idx))
            ],
            columns=my_explainer.feature_names,
        )
        for j, idx in zip(
            range(len(my_explainer.explanation.corpus_breakdown)),
            last_time_step_idx,
        )
    ]
    # Patient importances
    importance_dfs = [
        pd.DataFrame(
            my_explainer.explanation.feature_importances[j][
                idx - (time_steps_to_display - 1) : idx + 1
            ],
            index=[
                f"(t_max) - {i}" if i != 0 else "(t_max)"
                for i in reversed(range(time_steps_to_display))
            ],
            columns=[f"{col}_fi" for col in my_explainer.feature_names],
        )
        if time_steps_to_display <= idx
        else pd.DataFrame(
            my_explainer.explanation.feature_importances[j][0 : idx + 1],
            index=[
                f"(t_max) - {i}" if i != 0 else "(t_max)" for i in reversed(range(idx))
            ],
            columns=[f"{col}_fi" for col in my_explainer.feature_names],
        )
        for j, idx in zip(
            range(len(my_explainer.explanation.feature_importances)),
            last_time_step_idx,
        )
    ]

    corpus_data = [
        {
            "feature_vals": corpus_dfs[i].transpose(),
            "Label": simplex_explainer.apply_sort_order(
                my_explainer.corpus_targets, my_explainer.explanation.sort_order
            )[i],
            "Prediction": simplex_explainer.apply_sort_order(
                my_explainer.corpus_predictions,
                my_explainer.explanation.sort_order,
            )[i],
            "Example Importance": my_explainer.explanation.corpus_importances[i],
        }
        for i in range(len(corpus_dfs))
    ]
    importance_data = [
        {
            "importance_vals": importance_dfs[i].transpose(),
            "Label": simplex_explainer.apply_sort_order(
                my_explainer.corpus_targets, my_explainer.explanation.sort_order
            )[i],
            "Prediction": simplex_explainer.apply_sort_order(
                my_explainer.corpus_predictions,
                my_explainer.explanation.sort_order,
            )[i],
            "Example Importance": my_explainer.explanation.corpus_importances[i],
        }
        for i in range(len(corpus_dfs))
    ]

    max_importance = max([example["Example Importance"] for example in corpus_data])
    corpus_data = [
        example
        for example in corpus_data
        if example["Example Importance"] >= example_importance_threshold
    ]
    importance_data = [
        example
        for example in importance_data
        if example["Example Importance"] >= example_importance_threshold
    ]

    # Write corpus to app
    corpus_col1, corpus_col2 = st.columns(2)
    corpus_cols = [corpus_col1, corpus_col2]
    for example_i in range(len(corpus_data)):
        importance_df_colors = simplex_explainer.df_values_to_colors(
            importance_data[example_i]["importance_vals"].copy(),
            exclude_trailing_n_cols=0,
        )
        importance_df_colors = importance_df_colors.applymap(
            lambda x: f"background-color: {x}"
        )
        display_corpus_df = corpus_data[example_i]["feature_vals"].style.apply(
            highlight, axis=None
        )
        with corpus_cols[example_i % 2]:
            st.write(f"Corpus Example: {example_i}")
            st.write(
                f"Example Importance: {100 * corpus_data[example_i]['Example Importance']:0.2f}%"
            )
            st.write(display_corpus_df)


# Title and Intro
st.write("# SimplEx")
st.write(
    """
SimplEx is a case-based interpretability method. It can work with either tabular or time series data. You can read more about it in the [paper](https://papers.nips.cc/paper/2021/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html).

For clinically focussed examples go to the bespoke [SimplEx Demonstrator](https://vanderschaarlab-demo-simplex-simplexdemoapp-rjaur5.streamlitapp.com/). 
And for further information, [here](https://youtu.be/it-nfwPt4B8?t=6512) is a video demonstration of the clinical SimplEx app.
"""
)

preloaded_tab, upload_tab = st.tabs(["Examples", "Upload your own Explainer"])
with preloaded_tab:
    # Select simplEx data type, dataset, and model
    select_box_col1, select_box_col2, select_box_col3, *other_cols = st.columns(5)
    with select_box_col1:
        data_type = st.selectbox(
            label="Data type:",
            options=["Tabular", "Time Series"],
            key="data_type_select",
        )
    with select_box_col2:
        dataset_options = (
            ["iris", "wine"] if data_type == "Tabular" else ["Engine Noise"]
        )
        dataset = st.selectbox(
            label="Dataset:",
            options=dataset_options,
            key="dataset_select",
        )
    with select_box_col3:
        model_options = (
            ["MLP", "Linear"] if data_type == "Tabular" else ["Convolutional Net"]
        )
        model = st.selectbox(
            label="Model:",
            options=model_options,
            key="model_select",
        )

    slider_col1, slider_col2, *other_slider_col3 = st.columns(3)
    with slider_col1:
        test_example_id = st.slider("Test record:", 0, 20, 0, key="example_test_slider")

    # SimplEx
    if data_type == "Tabular":
        simplex_paths = {
            "MLP": {
                "iris": "resources/saved_explainers/simplex/tabular/iris_mlp_simplex_explainer_4.p",
                "wine": "resources/saved_explainers/simplex/tabular/wine_mlp_simplex_explainer_4.p",
            },
            "Linear": {
                "iris": "resources/saved_explainers/simplex/tabular/iris_linear_simplex_explainer_4.p",
                "wine": "resources/saved_explainers/simplex/tabular/wine_linear_simplex_explainer_4.p",
            },
        }
    else:
        simplex_paths = {
            "Convolutional Net": {
                "Engine Noise": "resources/saved_explainers/simplex/temporal/forda_conv_time_simplex_explainer_4.p",
            },
            "GRU": {
                "Engine Noise": "resources/saved_explainers/simplex/temporal/forda_gru_time_simplex_explainer.p",
            },  # TODO: Train this model and save the explainer before implementing
        }
    my_explainer = io.load_explainer(simplex_paths[model][dataset])

    my_explainer.explain(test_example_id, baseline="median")

    if data_type == "Tabular":
        explain_record_df, display_corpus_df = my_explainer.summary_plot(
            output_file_prefix="my_output",
            open_in_browser=False,
            return_type="styled_df",
        )
        for col in explain_record_df.columns:
            if col not in ["Test Prediction", "Test Label"]:
                explain_record_df[col] = explain_record_df[col].astype(float)
        st.write("### Test record:")
        st.write(explain_record_df)
        st.write("### Corpus:")
        st.write(display_corpus_df)

    elif data_type == "Time Series":
        write_time_series_explaination_to_app(my_explainer)
    # Display code in expander
    with st.expander("See code"):
        if data_type == "Tabular":
            st.write("### Implementation of the SimplexTabularExplainer")
            st.code(
                """
    # Initialize SimplEX, fit it on test examples
    my_explainer = simplex_explainer.SimplexTabularExplainer(
        model,
        X_corpus,
        y_corpus,
        feature_names=feature_names,
        corpus_size=corpus_size,
    )
    my_explainer.fit(X_explain, y_explain)

    # Explain 0th test example
    test_example_idx = 0
    explanation = my_explainer.explain(
        test_example_idx,
        baseline="median",
    )

    # Plot explanation
    my_explainer.summary_plot(
        output_file_prefix="my_output",
        open_in_browser=True,
    )

            """
            )
        if data_type == "Time Series":
            st.write("### Implementation of the SimplexTimeSeriesExplainer")
            st.code(
                """
    # Initialize SimplEX, fit it on test examples
    my_explainer = simplex_explainer.SimplexTimeSeriesExplainer(
        model,
        X_corpus,
        y_corpus,
        corpus_size=corpus_size,
    )
    my_explainer.fit(X_explain, y_explain)

    # Explain 0th test example
    test_example_idx = 0
    result, sort_order = my_explainer.explain(test_example_idx, baseline="median")

    # Plot explanation
    my_explainer.summary_plot(
        rescale_dict=rescale_dict,
        example_importance_threshold=0.15,
        time_steps_to_display=10,
        output_file_prefix="my_output",
        open_in_browser=True,
    )
            """
            )

# The Upload tab
with upload_tab:
    # Display code in expander
    with st.expander("How to create and upload your explainer"):
        st.write("### Tabular Data")
        st.write("")
        st.code(
            """
# Import Explainer
from interpretability import interpretability_models

from interpretability_models import simplex_explainer
from interpretability_models.utils import io

# Initialize SimplEX Explainer
my_explainer = simplex_explainer.SimplexTabularExplainer(
    model, # black-box model
    X_corpus, # Corpus example inputs (can be pandas DataFrame)
    y_corpus, # Corpus example targets (can be pandas Series)
    feature_names=feature_names, # List of feature names
    corpus_size=corpus_size, # integer corpus size
)

# Fit explainer on test examples
my_explainer.fit(
    X_explain, # Data to explain inputs (can be pandas DataFrame)
    y_explain, # Data to explain targets (can be pandas Series)
)

# Save explainer to file
io.save_explainer(explainer, save_path)
        """
        )

        st.write("*Then, if required move the file to the pick-up folder.*")
        st.write("### Time Series Data")
        st.write(
            "The same code applies, but please use the SimplexTimeSeriesExplainer class instead of the SimplexTabularExplainer."
        )
        st.write("### Upload")
        st.write(
            "You can then drop the resulting pickle file into the upload area below."
        )

    tabular_tab, time_series_tab = st.tabs(["Tabular", "Time Series"])
    #
    with tabular_tab:
        uploaded_explainer = st.file_uploader(
            "Upload explainer:", key="tabular_explainer_uploader"
        )
        if uploaded_explainer:

            my_explainer = pkl.load(uploaded_explainer)

        baseline_box_col1, slider_col, *other_cols = st.columns(5)
        with baseline_box_col1:
            baseline = st.selectbox(
                label="Baseline:",
                options=[
                    "zeros",
                    "constant",
                    "median",
                    "mean",
                ],
                key="baseline_tabular",
            )
            constant_val = 0
            if baseline == "constant":
                constant_val = st.number_input(
                    label="Baseline constant value:",
                    key="baseline_constant_value",
                )
        with slider_col:
            test_example_id = st.slider(
                "Test record:",
                0,
                my_explainer.explain_inputs.shape[0] - 1,
                0,
                key="tabular_explainer_slider",
            )

        if uploaded_explainer:
            my_explainer.explain(
                test_example_id, baseline=baseline, constant_val=constant_val
            )
            try:
                explain_record_df, display_corpus_df = my_explainer.summary_plot(
                    output_file_prefix="my_output",
                    open_in_browser=False,
                    return_type="styled_df",
                    output_html=False,
                )
                for col in explain_record_df.columns:
                    if col not in ["Test Prediction", "Test Label"]:
                        explain_record_df[col] = explain_record_df[col].astype(float)

                st.write("### Test record:")
                st.write(explain_record_df)

                st.write("### Corpus:")
                st.write(display_corpus_df)
            except:
                st.write("Are you sure this is a tabular explainer?")
    with time_series_tab:
        uploaded_explainer = st.file_uploader(
            "Upload explainer:", key="time_explainer_uploader"
        )
        baseline_box_col1, slider_col, *other_cols = st.columns(5)
        with baseline_box_col1:
            baseline = st.selectbox(
                label="Baseline:",
                options=[
                    "zeros",
                    "constant",
                    "median",
                    "mean",
                ],
                key="baseline_time",
            )
            constant_val = 0
            if baseline == "constant":
                constant_val = st.number_input(
                    label="Baseline constant value:",
                    key="baseline_constant_value",
                )
        with slider_col:
            test_example_id = st.slider(
                "Test record:",
                0,
                my_explainer.explain_inputs.shape[0] - 1,
                0,
                key="time_explainer_slider",
            )

        if uploaded_explainer:

            my_explainer = pkl.load(uploaded_explainer)
            my_explainer.explain(
                test_example_id, baseline=baseline, constant_val=constant_val
            )
            write_time_series_explaination_to_app(my_explainer)
