import streamlit as st
from streamlit_extras import switch_page_button

st.set_page_config(
    page_title="Interpretability Suite",
    page_icon="images/ccaim_logo-no_text.png",
    layout="wide",
)

# st.image("images/Explain_image.jpg")
st.write("# Interpretability Suite")

st.sidebar.success("Select an interpretability method above.")

st.write(
    """
    This is the Interpretability Suite developed by CCAIM and the van der schaar lab.

    This app is a collection of different Machine Learning interpretability methods and aims to
    show you how these methods work, what they can do for you and how you can to implement them yourself.
    It includes methods from the van der Schaar Lab, alongside other key methods from the community.
    This is a work in progress with more methods potentially to be added so for an exhaustive list of the
    methods, please check out the [README](https://github.com/vanderschaarlab/Interpretability) file on the github.
    There you will find further guidance to help you select the method that is right for you! Or for further info,
    this [video](https://www.youtube.com/watch?v=R-27AiRK1r0) offers a brief introduction to the methods in the GitHub repository.

    All the methods aim to provide an insight into why a machine learning model has made a given prediction.
    This is critical because for a model's predictions to be trusted they must be understood.

    **Please select one of our methods** to see some examples
    of what interpretability could do for your black-box models!

"""
)

button_col1, button_col2, button_col3, button_col4, *other_cols = st.columns(9)
with button_col1:
    if st.button(
        "SimplEx",
        key="simplex_navigation_button",
    ):
        switch_page_button.switch_page("SimplEx")
with button_col2:
    if st.button(
        "Dynamask",
        key="dynamask_navigation_button",
    ):
        switch_page_button.switch_page("Dynamask")
with button_col3:
    if st.button(
        "Shap",
        key="shap_navigation_button",
    ):
        switch_page_button.switch_page("Shap")
with button_col4:
    if st.button(
        "Symbolic Pursuit",
        key="symbolic_pursuit_navigation_button",
    ):
        switch_page_button.switch_page("Symbolic_Pursuit")

st.write("")
st.write("### Method Selection")

st.image(
    "images/Interpretability_method_flow_diagram.svg",
    width=10,
)
st.write("*Fig 1: A flow chart to aid interpretability method selection.*")
st.markdown("""---""")
st.write("*This app has been produced by*")
st.image("images/ccaim_logo_white_background.png", width=400)
