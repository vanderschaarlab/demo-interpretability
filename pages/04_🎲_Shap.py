# Imports
# Standard
import sys
import os
import streamlit as st
import pandas as pd
import torch
import dill as pkl

# Interpretability
from interpretability.interpretability_models.utils import data, io

# from models.multilayer_perceptron import IrisMLP

st.set_page_config(
    page_title="Interpretability Suite",
    page_icon="images/ccaim_logo-no_text.png",
    layout="wide",
)
st.write("# Shap")
st.write(
    'Shap uses "Shapley values" to indicate relative feature importance. It is a widely used approach derived from cooperative game theory that was developed externally from the lab. Shap is compatible with many different model types.'
)

preloaded_tab, upload_tab = st.tabs(["Examples", "Upload your own Explainer"])
with preloaded_tab:
    # Select data type, dataset, and model
    select_box_col1, select_box_col2, select_box_col3, *other_cols = st.columns(5)
    with select_box_col1:
        model_type = st.selectbox(
            label="Model type:",
            # options=["Linear", "Tree", "Deep", "Kernel"],
            options=["Deep", "Kernel"],
            key="model_type_select",
        )
    with select_box_col2:
        dataset_options = (
            ["iris", "wine"] if model_type != "Kernel" else ["iris"]
        )  # remove if after re-train kernel
        dataset = st.selectbox(
            label="Dataset:",
            options=dataset_options,
            key="dataset_select",
        )
    if dataset == "iris":
        with st.expander("Explanation of iris data:"):
            st.write(
                "This is a classification task with three possible output classes. The task is the identification of species of iris plant, based on petal/sepal measurements."
            )
            st.table(
                data={
                    "Feature name": [
                        "sepal length (cm)",
                        "sepal width (cm)",
                        "petal length (cm)",
                        "petal width (cm)",
                    ],
                    "[min, max] value in dataset": [
                        "[4.3, 7.9]",
                        "[2.0, 4.4]",
                        "[1.0, 6.9]",
                        "[0.1, 2.5]",
                    ],
                    "median values": [5.8, 3.0, 4.35, 1.3],
                }
            )
    if dataset == "wine":
        with st.expander("Explanation of wine quality data:"):
            st.write(
                "This is a classification task where the output is a score out of 10 for the quality of the wine. The majority of wines have a score between 3 and 9."
            )
            st.table(
                data={
                    "Feature name": [
                        "fixed acidity",
                        "volatile acidity",
                        "citric acid",
                        "residual sugar",
                        "chlorides",
                        "free sulfur dioxide",
                        "total sulfur dioxide",
                        "density",
                        "pH",
                        "sulphates",
                        "alcohol",
                    ],
                    "[min, max] value in dataset": [
                        "[3.8, 15.9]",
                        "[0.08, 1.58]",
                        "[0, 1.66]",
                        "[0.6, 65.8]",
                        "[0.009, 0.611]",
                        "[1, 289]",
                        "[6, 440]",
                        "[0.987, 1.039]",
                        "[2.72, 4.01]",
                        "[0.22, 2]",
                        "[8, 14.9]",
                    ],
                    "median values": [
                        7,
                        0.29,
                        0.31,
                        3,
                        0.047,
                        29,
                        118,
                        0.995,
                        3.21,
                        0.51,
                        10.3,
                    ],
                }
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

        my_shap_explainer = io.load_explainer(shap_paths[model_type][dataset])

        # my_shap_explainer.explain()

        my_shap_explainer.summary_plot(
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
        my_shap_explainer = pkl.load(uploaded_explainer)

        my_shap_explainer.summary_plot(
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
