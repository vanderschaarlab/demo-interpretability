# Imports
# Standard
import sys
import re
import os
import math

# Third party
import streamlit as st
import numpy as np
import dill as pkl
import sympy as smp
import textwrap
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Interpretability
from interpretability.interpretability_models.utils import data, io

st.set_page_config(
    page_title="Interpretability Suite",
    page_icon="images/ccaim_logo-no_text.png",
    layout="wide",
)


def render_output(
    my_symbolic_pursuit_explainer, suffix="preloaded", scale_with_preloaded_scaler=None
):
    def round_coefficients_in_str_expression(str_expression, decimal_places):
        coeffs = re.findall(r"\d+\.\d{4,15}", str_expression)
        short_coeffs = [str(round(float(c), decimal_places)) for c in coeffs]
        coeff_repl = zip(coeffs, short_coeffs)
        for c, s_c in coeff_repl:
            str_expression = re.sub(c, s_c, str_expression, count=1)
        return str_expression

    def df_values_to_colors(df):
        """Gets color values based in values relative to all other values in df."""
        min_val = np.nanmin(df.values)
        max_val = np.nanmax(df.values)

        for col in df:
            # map values to colors in hex via
            # creating a hex Look up table table and apply the normalized data to it
            norm = mcolors.Normalize(
                vmin=min_val,
                vmax=max_val,
                clip=True,
            )
            lut = plt.cm.bwr(np.linspace(0.2, 0.75, 256))
            lut = np.apply_along_axis(mcolors.to_hex, 1, lut)
            a = (norm(df[col].values) * 255).astype(np.int16)
            df[col] = lut[a]
        return df

    def highlight(x):
        return pd.DataFrame(
            importance_df_colors.values, index=x.index, columns=x.columns
        )

    def create_coefficient_heatmap_from_second_order_taylor_expansion(
        expression, feature_names
    ):
        expression = smp.Poly(expression)
        symbols = list(expression.free_symbols)
        symbol_pairs = itertools.product(symbols, repeat=2)
        coeffs_dict = {}
        for s_p in symbol_pairs:
            coeffs_dict[f"{s_p[0]}{s_p[1]}"] = expression.coeff_monomial(
                s_p[0] * s_p[1]
            )
        coeffs_dict = dict(sorted(coeffs_dict.items()))
        coeffs_dict_reoriented = {}
        for i in range(num_features):
            coeffs_dict_reoriented[f"{feature_names[i]}"] = [
                float(coeffs_dict[f"X{i}X{j}"]) for j in range(num_features)
            ]
        coeffs = pd.DataFrame(data=coeffs_dict_reoriented, index=feature_names)
        mask = np.triu(coeffs)
        np.fill_diagonal(
            mask, 0
        )  # show main diagonal, to allow comparison of the interaction importances to the features alone
        plt.clf()
        figure = sns.heatmap(
            coeffs, annot=True, mask=mask, fmt=".2f", annot_kws={"fontsize": 4}
        ).get_figure()
        st.pyplot(figure)

    st.write("## Symbolic Expression of the model")
    st.write(
        "This is the explicit expression that has been learnt through the fitting process. It approximates the predictive model."
    )
    symb_expression_str = smp.latex(
        my_symbolic_pursuit_explainer.explanation.expression
    )
    symb_expression_terms = round_coefficients_in_str_expression(
        symb_expression_str, 3
    ).split("+")
    for i, term in enumerate(symb_expression_terms):
        if i == 0:
            st.latex(term)
        else:
            st.latex("+ " + term)

    with st.expander("Projections in the Symbolic Expression:"):
        st.write(
            "These are the symbolic projections (P₁ to Pₙ) in the expression above."
        )
        for i, projection in enumerate(
            my_symbolic_pursuit_explainer.symbolic_model.get_projections()
        ):
            proj_str = smp.latex(projection)
            proj_str = round_coefficients_in_str_expression(proj_str, 3)
            proj_lines = textwrap.fill(proj_str, 180)

            for line_idx, p_line in enumerate(proj_lines.split("\n")):
                if line_idx == 0:
                    st.latex(rf"""P_{i+1} = {p_line}""")
                else:
                    st.latex(p_line)

    st.write("## Symbolic Model to Predictions")
    st.write("**Inputs**")
    num_features = my_symbolic_pursuit_explainer.symbolic_model.dim_x
    number_of_feature_cols = 10
    cols = st.columns(num_features)
    for i in range(len(cols)):
        with cols[i % number_of_feature_cols]:
            if suffix == "preloaded":
                feature = st.number_input(
                    f"{my_symbolic_pursuit_explainer.feature_names[i]}",
                    value=median_values[i],
                    key=f"{my_symbolic_pursuit_explainer.feature_names[i]}_input_val_{suffix}",
                )  # Feature_names must be passed to the explainer to be used in this interface
            if suffix == "uploaded":
                feature = st.number_input(
                    f"{my_symbolic_pursuit_explainer.feature_names[i]}",
                    value=0.0,
                    key=f"{my_symbolic_pursuit_explainer.feature_names[i]}_input_val_{suffix}",
                )  # no median values known for uploaded explainer, so default to 0

    inputs_to_predict = np.array(
        [
            st.session_state[
                f"{my_symbolic_pursuit_explainer.feature_names[i]}_input_val_{suffix}"
            ]
            for i in range(num_features)
        ]
    ).reshape(1, num_features)

    if scale_with_preloaded_scaler:
        with open(f"resources/data_scalers/{dataset}_scaler.p", "rb") as f:
            scaler = pkl.load(f)
        inputs_to_predict = scaler.transform(inputs_to_predict)

    st.write("**Predictions**")
    if dataset == "iris":
        st.write(
            f"""
            Symbolic Model Prediction: {int(round(my_symbolic_pursuit_explainer.symbolic_model.predict(inputs_to_predict).item(0),0))}
        """
        )
    else:
        st.write(
            f"""
            Symbolic Model Prediction: {round(my_symbolic_pursuit_explainer.symbolic_model.predict(inputs_to_predict).item(0),2)}
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
        Predictive Model Prediction: {round(predictive_model_pred, 4)}
    """
    )
    if dataset == "diabetes":
        st.write(
            "This output value is a quantitative measure of disease progression one year after baseline. The minimum value in the dataset is 25.0. The maximum value in the dataset is 346.0"
        )
    st.write("## Feature Importance")
    st.write(
        "The following are feature importances for the given input. Changing the imput with the +/- buttons will update the importances."
    )

    feature_importance_zip = zip(
        my_symbolic_pursuit_explainer.feature_names,
        my_symbolic_pursuit_explainer.symbolic_model.get_feature_importance(
            inputs_to_predict.reshape(num_features)
        ),
    )
    feature_importance_dict = {
        k: [round(float(v), 2)] for k, v in feature_importance_zip
    }
    feature_importance_df = pd.DataFrame(data=feature_importance_dict)

    importance_df_colors = df_values_to_colors(
        feature_importance_df.copy(),
    )
    importance_df_colors = importance_df_colors.applymap(
        lambda x: f"background-color: {x}"
    )
    table_width = 25
    if len(feature_importance_df.columns) <= table_width:
        st.write(feature_importance_df.style.apply(highlight, axis=None))
    else:
        feature_importance_dfs = np.array_split(
            feature_importance_df,
            math.ceil(len(feature_importance_df.columns) / table_width),
            axis=1,
        )
        for feature_importance_df in feature_importance_dfs:
            st.write(feature_importance_df)

    st.write("## Feature Interactions")
    st.write(
        "This heatmap shows the feature interactions in the symbolic expression. Values with the highest magnitude, positive or negative, show the feature interactions with the largest effect on the predicted value. It is calculated by examining the coefficents in the second order Taylor expansion of the learned symbolic expression around the input point given above."
    )
    taylor_expand_expr = smp.expand(
        my_symbolic_pursuit_explainer.symbolic_model.get_taylor(
            inputs_to_predict.reshape(num_features), 2
        )
    )
    try:
        create_coefficient_heatmap_from_second_order_taylor_expansion(
            taylor_expand_expr, my_symbolic_pursuit_explainer.feature_names
        )
    except:
        st.write("Please ensure that you have selected valid inputs.")


### Main
st.write("# Symbolic Pursuit Explainer")
st.write(
    "Symbolic Pursuit is an interpretability method, where the Black-Box is approximated by a closed-form mathematical expression. It is a variation of projection pursuit that allows for global explanations. You can read more about it in the [paper](https://arxiv.org/abs/2011.08596#:~:text=Learning%20outside%20the%20Black%2DBox%3A%20The%20pursuit%20of%20interpretable%20models,-Jonathan%20Crabb%C3%A9%2C%20Yao&text=Machine%20Learning%20has%20proved%20its,difficulties%20of%20interpreting%20these%20models.)."
)

preloaded_tab, upload_tab = st.tabs(["Examples", "Upload your own Explainer"])

with preloaded_tab:
    # Select boxes to choose explainer
    select_box_col1, select_box_col2, *other_cols = st.columns(5)
    with select_box_col1:
        dataset_options = ["iris", "diabetes"]
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
            "iris": "resources/saved_explainers/symbolic_pursuit/iris_sklearn_random_forrest_explainer_regression.p",
        },
        "Linear": {
            "diabetes": "resources/saved_explainers/symbolic_pursuit/diabetes_sklearn_linear_explainer_6.p",
            "iris": "resources/saved_explainers/symbolic_pursuit/iris_sklearn_linear_explainer_regression.p",
        },
        "Multi-Layer Perceptron": {
            "diabetes": "resources/saved_explainers/symbolic_pursuit/diabetes_sklearn_mlp_explainer_7.p",
            "iris": "resources/saved_explainers/symbolic_pursuit/iris_sklearn_mlp_explainer_regression.p",
        },
    }
    loaded_symbolic_pursuit_explainer = io.load_explainer(
        sym_pursuit_paths[model][dataset]
    )

    loaded_symbolic_pursuit_explainer.explain()

    if dataset == "diabetes":
        median_values = [50, 1, 25.7, 93, 186, 113, 48, 4, 4.62, 91]
        scale_with_preloaded_scaler = True
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
    elif dataset == "iris":
        median_values = [5.8, 3.0, 4.35, 1.3]
        scale_with_preloaded_scaler = False
        with st.expander("Explanation of iris data:"):
            st.write(
                "This is a classification task with the possible output classes of 1, 2, or 3."
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
                    "median values": median_values,
                }
            )

    render_output(
        loaded_symbolic_pursuit_explainer,
        suffix="preloaded",
        scale_with_preloaded_scaler=scale_with_preloaded_scaler,
    )

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
