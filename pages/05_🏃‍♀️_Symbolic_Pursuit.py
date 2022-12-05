# Imports
# Standard
import sys
import re
import os

# Third party
import streamlit as st
import numpy as np
import dill as pkl
import sympy as smp
import textwrap
import torch

# Interpretability
from interpretability.interpretability_models.utils import data, io

st.set_page_config(
    page_title="Interpretability Suite",
    page_icon="images/ccaim_logo-no_text.png",
    layout="wide",
)


def render_output(my_symbolic_pursuit_explainer, suffix="preloaded"):
    st.write("### Symbolic Pursuit Output")
    st.write(
        "The symbolic model has approximated the model with the following expression. This expression includes the projections which has weights associated with each of the features."
    )

    st.write("**Symbolic Expression of the model:**")
    if suffix == "preloaded":
        symb_expression_terms = smp.latex(
            my_symbolic_pursuit_explainer.explanation.expression
        ).split("+")
        for i, term in enumerate(symb_expression_terms):
            if i == 0:
                st.latex(term)
            else:
                st.latex("+ " + term)
    else:
        st.write(my_symbolic_pursuit_explainer.explanation.expression)

    st.write("**Projections in the Symbolic Expression:**")
    for i, projection in enumerate(
        my_symbolic_pursuit_explainer.symbolic_model.get_projections()
    ):
        proj_lines = textwrap.fill(smp.latex(projection), 180)
        for line_idx, p_line in enumerate(proj_lines.split("\n")):
            if line_idx == 0:
                st.latex(rf"""P_{i+1} = {p_line}""")
            else:
                st.latex(p_line)

    st.write("### Symbolic Model to Predictions and Feature Importance")
    st.write("**Inputs:**")
    num_features = my_symbolic_pursuit_explainer.symbolic_model.dim_x
    cols = st.columns(num_features)
    for i, col in enumerate(cols):
        with col:
            feature = st.number_input(
                f"{my_symbolic_pursuit_explainer.feature_names[i]}",
                value=median_values[i],
                key=f"{my_symbolic_pursuit_explainer.feature_names[i]}_input_val_{suffix}",
            )  # Feature_names must be passed to the explainer to be used in this interface
    inputs_to_predict = np.array(
        [
            st.session_state[
                f"{my_symbolic_pursuit_explainer.feature_names[i]}_input_val_{suffix}"
            ]
            for i in range(num_features)
        ]
    ).reshape(1, num_features)

    with open(f"resources/data_scalers/{dataset}_scaler.p", "rb") as f:
        scaler = pkl.load(f)
    inputs_to_predict = scaler.transform(inputs_to_predict)

    st.write("**Predictions:**")
    st.write(
        f"""
        Symbolic Model Prediction: {my_symbolic_pursuit_explainer.symbolic_model.predict(inputs_to_predict).item(0)}
    """
    )
    try:
        predictive_model_pred = my_symbolic_pursuit_explainer.model(
            inputs_to_predict
        ).item(0)
    except:
        predictive_model_pred = my_symbolic_pursuit_explainer.model(
            torch.Tensor(inputs_to_predict)
        ).item(0)
    st.write(
        f"""
        Predictive Model Prediction: {predictive_model_pred}
    """
    )
    if dataset == "diabetes":
        st.write(
            "This output value is a quantitative measure of disease progression one year after baseline. The minimum value in the dataset is 25.0. The maximum value in the dataset is 346.0"
        )

    st.write("**Feature Importance:**")
    st.write(
        "The following are feature importances for the given input. Changing the imput with the +/- buttons will update the importances."
    )
    fi_cols = st.columns(num_features)
    for i, col in enumerate(fi_cols):
        with col:
            st.write(
                f"{my_symbolic_pursuit_explainer.feature_names[i]}: {round(my_symbolic_pursuit_explainer.symbolic_model.get_feature_importance(inputs_to_predict.reshape(num_features))[i], 4)}"
            )
    st.write("### Second order Taylor expansion")
    st.write(
        "This is the second order Taylor expansion of the learned symbolic expression around the input point given above. The weights against the different pairwise feature products gives an indication of the feature interactions."
    )
    taylor_expand_lines = textwrap.fill(
        smp.latex(
            smp.expand(
                my_symbolic_pursuit_explainer.symbolic_model.get_taylor(
                    inputs_to_predict.reshape(num_features), 2
                )
            )
        ),
        180,
    )
    for line_idx, t_line in enumerate(taylor_expand_lines.split("\n")):
        st.latex(t_line)


st.write("# Symbolic Pursuit Explainer")
st.write(
    "Symbolic Pursuit is an interpretability method, where the Black-Box is approximated by a simple symbolic equation. It is a variation of projection pursuit that allows for global explanations. You can read more about it in the [paper](https://arxiv.org/abs/2011.08596#:~:text=Learning%20outside%20the%20Black%2DBox%3A%20The%20pursuit%20of%20interpretable%20models,-Jonathan%20Crabb%C3%A9%2C%20Yao&text=Machine%20Learning%20has%20proved%20its,difficulties%20of%20interpreting%20these%20models.)."
)

preloaded_tab, upload_tab = st.tabs(["Examples", "Upload your own Explainer"])

with preloaded_tab:
    # Select boxes to choose explainer
    select_box_col1, select_box_col2, *other_cols = st.columns(5)
    with select_box_col1:
        dataset_options = ["diabetes"]
        dataset = st.selectbox(
            label="Dataset:",
            options=dataset_options,
            key="dataset_select_preload",
        )
    with select_box_col2:
        model_options = ["Random Forrest", "Multi-Layer Perceptron", "Linear"]
        model = st.selectbox(
            label="Predictive Model:",
            options=model_options,
            key="model_select_preload",
        )
    # Load the explainer
    sym_pursuit_paths = {
        "Random Forrest": {
            "diabetes": "resources/saved_explainers/symbolic_pursuit/diabetes_sklearn_random_forrest_explainer.p",
        },
        "Linear": {
            "diabetes": "resources/saved_explainers/symbolic_pursuit/diabetes_sklearn_linear_explainer_6.p",
        },
        "Multi-Layer Perceptron": {
            "diabetes": "resources/saved_explainers/symbolic_pursuit/diabetes_sklearn_mlp_explainer_7.p",
        },
    }
    loaded_symbolic_pursuit_explainer = io.load_explainer(
        sym_pursuit_paths[model][dataset]
    )

    loaded_symbolic_pursuit_explainer.explain()

    if dataset == "diabetes":
        median_values = [50, 1, 25.7, 93, 186, 113, 48, 4, 4.62, 91]
        with st.expander("Explanation of diabetes data:"):
            st.table(
                data={
                    "Feature name": [
                        "age",
                        "sex",
                        "bmi",
                        "bp",
                        "s1",
                        "s2",
                        "s3",
                        "s4",
                        "s5",
                        "s6",
                    ],
                    "Explanation": [
                        "age in years",
                        "represented as an integer [1 or 2]",
                        "body mass index",
                        "average blood pressure",
                        "tc, total serum cholesterol",
                        "ldl, low-density lipoproteins",
                        "hdl, high-density lipoproteins",
                        "tch, total cholesterol / HDL",
                        "ltg, log of serum triglycerides level",
                        "glu, blood sugar level",
                    ],
                    "[min, max] value in dataset": [
                        "[19, 79]",
                        "[1, 2]",
                        "[18, 42.2]",
                        "[62, 133]",
                        "[97, 301]",
                        "[41, 242.4]",
                        "[22, 99]",
                        "[2, 9.09]",
                        "[3.26, 6.11]",
                        "[58, 124]",
                    ],
                    "median values": median_values,
                }
            )

    render_output(loaded_symbolic_pursuit_explainer, suffix="preloaded")

with upload_tab:

    uploaded_explainer = st.file_uploader(
        "Upload explainer:", key="symbolic_pursuit_explainer_uploader"
    )
    if uploaded_explainer:

        # Load the explainer
        my_symbolic_pursuit_explainer = pkl.load(uploaded_explainer)
        my_symbolic_pursuit_explainer.explain()
        render_output(my_symbolic_pursuit_explainer, suffix="uploaded")


with st.expander("See code"):
    st.write("Implementation of the SymbolicPursuitExplainer.")
    st.code(
        """
import numpy as np
from interpretability_models import symbolic_pursuit_explainer

my_explainer = symbolic_pursuit_explainer.SymbolicPursuitExplainer(
    model.predict, # The predictive model to explain
    X_train, # The data used to fit the SymbolicRegressor
    loss_tol=0.1, # The tolerance for the loss under which the pursuit stops. 
    patience=5, # A hard limit on the number of optimisation loops in fit().


my_explainer.fit()

my_explainer.measure_fit_quality(X_test, y_test) # Prints both the mean squared error for the predictive model and the Symbolic Regressor. 

explanation = my_explainer.explain() # Get the symbolic expression and projections

my_explainer.summary_plot()

    """
    )
