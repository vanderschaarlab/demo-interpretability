# Imports
# Standard
import sys
import os
import streamlit as st
import numpy as np
import dill as pkl
import torch

# Interpretability
from interpretability.interpretability_models.utils import data, io

st.set_page_config(
    page_title="Interpretability Suite",
    page_icon="images/ccaim_logo-no_text.png",
    layout="wide",
)


# @st.cache(hash_funcs={torch.Tensor: my_hash_func}) # TODO: handle tor torch.tensor caching issue by uncommenting and adding a my_hash_func
def refit(explainer, explain_id):
    explainer.refit(explain_id)


st.write("# Dynamask")
st.write(
    "Dynamask is a model for time series data. You can read more about it in the [paper](https://arxiv.org/abs/2106.05303)."
)

preloaded_tab, upload_tab = st.tabs(["Examples", "Upload your own Explainer"])

with preloaded_tab:
    # Select boxes to choose explainer
    select_box_col1, select_box_col2, *other_cols = st.columns(5)
    with select_box_col1:
        dataset_options = ["Basic Motions", "Engine Noise"]
        dataset = st.selectbox(
            label="Dataset:",
            options=dataset_options,
            key="dataset_select_preload",
        )
    with select_box_col2:
        model_options = ["Convolutional Net"]
        model = st.selectbox(
            label="Model:",
            options=model_options,
            key="model_select_preload",
        )

    if dataset == "Engine Noise":
        with st.expander("Explanation of Engine Noise data:"):
            st.write(
                "This is a time series classification task which comprises of audio data of a running engine at 500 time steps. The output is a score of 0 or 1 based on whether or not there is a fault with the engine."
            )
            st.table(
                data={
                    "Feature name": ["Engine Noise"],
                    "Time Steps": 500,
                    "[min, max] value in dataset": [
                        "[-4.618, 5.059]",
                    ],
                    "median values": [0],
                }
            )
    if dataset == "Basic Motions":
        with st.expander("Explanation of Basic Motions data:"):
            st.write(
                "This is an equal length multivariate time series classification problem. It consists of label 4 classes with 80 cases, each with 100 time steps."
            )
            st.table(
                data={
                    "Feature name": ["Basic Motions"],
                    "Time Steps": 100,
                    "[min, max] value in dataset": [
                        "[27.822042, 34.86621]",
                    ],
                    "median values": [0],
                }
            )
    # Load the explainer
    dynamask_paths = {
        "Convolutional Net": {
            "Engine Noise": "resources/saved_explainers/dynamask/forda_conv_dynamask_explainer_4.p",
            "Basic Motions": "resources/saved_explainers/dynamask/basic_motions_conv_dynamask_explainer.p",
        },
    }
    my_dynamask_explainer = io.load_explainer(dynamask_paths[model][dataset])

    # Test examples to fit
    explain_id = st.slider(
        "Test record:",
        0,
        my_dynamask_explainer.all_data.shape[0] - 1,
        0,
        key="explain_id_slider_preload",
    )
    refit(my_dynamask_explainer, explain_id)

    # Parameters for explain
    slider_col1, slider_col2, *other_cols = st.columns(4)
    with slider_col1:
        features_displayed_start, features_displayed_stop = st.slider(
            "Features Displayed:",
            0,
            my_dynamask_explainer.explain_data.shape[1],
            (0, my_dynamask_explainer.explain_data.shape[1]),
            key="features_displayed_slider_preload",
        )
    with slider_col2:
        times_displayed_start, times_displayed_stop = st.slider(
            "Time steps Displayed:",
            0,
            my_dynamask_explainer.explain_data.shape[0],
            (0, my_dynamask_explainer.explain_data.shape[0]),
            key="times_displayed_slider_preload",
        )
    smooth_mask = st.checkbox(
        "Smooth the mask",
        value=False,
        key="smooth_mask_checkbox_preload",
        help="This smooths the mask tensor by applying a temporal Gaussian filter for each feature.",
    )

    my_dynamask_explainer.explain(
        ids_feature=list(range(features_displayed_start, features_displayed_stop)),
        ids_time=list(range(times_displayed_start, times_displayed_stop)),
        smooth=smooth_mask,
        get_mask_from_group_method="extremal",
        extremal_mask_threshold=0.01,
    )

    my_dynamask_explainer.summary_plot(
        my_dynamask_explainer.explanation,
        show=False,
        save_path="resources/saved_explainers/dynamask/temp_dynamask_plot.png",
    )
    temp_output = os.path.abspath(
        "resources/saved_explainers/dynamask/temp_dynamask_plot.png"
    )
    st.image(temp_output)

with upload_tab:

    uploaded_explainer = st.file_uploader(
        "Upload explainer:", key="dynamask_explainer_uploader"
    )
    if uploaded_explainer is not None:
        # Load the explainer
        my_dynamask_explainer = pkl.load(uploaded_explainer)

        # Test examples to fit
        explain_id = st.slider(
            "Test record:",
            0,
            my_dynamask_explainer.all_data.shape[0],
            0,
            key="explain_id_slider_upload",
        )

        refit(my_dynamask_explainer, explain_id)
        # Parameters for explain
        slider_col1, slider_col2, *other_cols = st.columns(4)
        with slider_col1:
            features_displayed = st.slider(
                "Features Displayed:",
                0,
                my_dynamask_explainer.explain_data.shape[1],
                my_dynamask_explainer.explain_data.shape[1],
                key="features_displayed_slider_upload",
            )
        with slider_col2:
            times_displayed = st.slider(
                "Time steps Displayed:",
                0,
                my_dynamask_explainer.explain_data.shape[0],
                my_dynamask_explainer.explain_data.shape[0],
                key="times_displayed_slider_upload",
            )
        smooth_mask = st.checkbox(
            "Smooth the mask",
            value=False,
            key="smooth_mask_checkbox_upload",
            help="This smooths the mask tensor by applying a temporal Gaussian filter for each feature.",
        )

        my_dynamask_explainer.explain(
            ids_feature=list(range(features_displayed)),
            ids_time=list(range(times_displayed)),
            smooth=smooth_mask,
            get_mask_from_group_method="extremal",
            extremal_mask_threshold=0.01,
        )

        my_dynamask_explainer.summary_plot(
            my_dynamask_explainer.explanation,
            show=False,
            save_path="resources/saved_explainers/dynamask/temp_dynamask_plot.png",
        )
        temp_output = os.path.abspath(
            "resources/saved_explainers/dynamask/temp_dynamask_plot.png"
        )
        st.image(temp_output)

with st.expander("See code"):
    st.write("Implementation of the DynamaskExplainer.")
    st.code(
        """
import numpy as np
from interpretability_models import dynamask_explainer

# Initialise the explainer
my_explainer = dynamask_explainer.DynamaskExplainer(
    model, "gaussian_blur", group=False
)

# Fit the explainer
my_explainer.fit(
    X_train[1],
    loss_function="mse",
    target=y_train,
    area_list=np.arange(0.1, 0.5, 0.1),
)
feature_num, time_step_num = X_train[0].shape

# Explain
my_explainer.explain(
    ids_feature=[i for i in range(5)],
    ids_time=[i for i in range(5)],
    smooth=False,
    get_mask_from_group_method="extremal",
    extremal_mask_threshold=0.01,
)

    """
    )
