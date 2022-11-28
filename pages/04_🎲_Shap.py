# Imports
import sys
import os
import streamlit as st
import pandas as pd
import torch
import dill as pkl

from interpretability.interpretability_models.utils import data, io

# from models.multilayer_perceptron import IrisMLP

st.set_page_config(
    page_title="Interpretability Suite",
    page_icon="images/ccaim_logo-no_text.png",
    layout="wide",
)
st.write("# Shap")
st.write("Shap is a method for various model types.")

preloaded_tab, upload_tab = st.tabs(["Examples", "Upload your own Explainer"])
with preloaded_tab:
    # Select simplEx data type, dataset, and model
    select_box_col1, select_box_col2, select_box_col3, *other_cols = st.columns(5)
    with select_box_col1:
        model_type = st.selectbox(
            label="Data type:",
            options=["Kernel", "Linear", "Tree", "Deep"],
            key="model_type_select",
        )
    with select_box_col2:
        dataset_options = ["iris", "wine"]
        dataset = st.selectbox(
            label="Dataset:",
            options=dataset_options,
            key="dataset_select",
        )

    # Shap
    shap_paths = {
        # "Deep": {
        #     "iris": "resources/saved_explainers/shap/iris_shap_deep_explainer.p",
        #     "wine": "resources/saved_explainers/shap/wine_simplex_explainer.p",
        # },
        "Linear": {
            "iris": "resources/saved_explainers/shap/iris_shap_linear_explainer_4.p",
            "wine": "resources/saved_explainers/shap/wine_shap_linear_explainer_4.p",
        },
        "Kernel": {
            "iris": "resources/saved_explainers/shap/iris_shap_kernel_explainer_4.p",
            "wine": "resources/saved_explainers/shap/wine_shap_kernel_explainer_4.p",
        },
        "Tree": {
            "iris": "resources/saved_explainers/shap/iris_shap_tree_explainer_4.p",
            "wine": "resources/saved_explainers/shap/wine_shap_tree_explainer_4.p",
        },
    }

    if model_type == "Deep":
        st.write(
            "The Deep SHAP model in non-serializable. You therefore cannot save or load Deep Shap explainers and cannot upload them to the 'Upload your own Explainer' tab."
        )
        if dataset == "iris":
            st.image(
                os.path.abspath(
                    "resources/saved_explainers/shap/iris_deep_shap_plot.png"
                )
            )
        elif dataset == "wine":
            st.image(
                os.path.abspath(
                    "resources/saved_explainers/shap/wine_deep_shap_plot.png"
                )
            )
    else:

        my_explainer = io.load_explainer(shap_paths[model_type][dataset])

        # my_explainer.explain()

        my_explainer.summary_plot(
            show=False, save_path="resources/saved_explainers/shap/temp_shap_plot.png"
        )

        temp_output_path = os.path.abspath(
            "resources/saved_explainers/shap/temp_shap_plot.png"
        )
        st.image(temp_output_path)

with upload_tab:
    uploaded_explainer = st.file_uploader(
        "Upload explainer:", key="shap_explainer_uploader"
    )
    if uploaded_explainer:
        # Load the explainer
        my_explainer = pkl.load(uploaded_explainer)

        my_explainer.summary_plot(
            show=False, save_path="resources/saved_explainers/shap/temp_shap_plot.png"
        )

        temp_output_path = os.path.abspath(
            "resources/saved_explainers/shap/temp_shap_plot.png"
        )
        st.image(temp_output_path)


with st.expander("See code"):
    if model_type == "Deep":

        st.write("Implementation of the DynamaskExplainer.")
        st.code(
            """
# Initialize shap with test examples
my_explainer = shap_explainer.ShapDeepExplainer(model, X_explain, y_explain)

# Explain
explanation = my_explainer.explain()
        """
        )

    if model_type == "Tree":

        st.write("Implementation of the DynamaskExplainer.")
        st.code(
            """
# Initialize shap with test examples
my_explainer = shap_explainer.ShapTreeExplainer(model, X_explain)
# Explain
explanation = my_explainer.explain()
        """
        )
