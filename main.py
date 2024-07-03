import streamlit as st
from PIL import Image

from apps.home import home_page
from apps.heart import heart_page
from apps.tb import tb_page
from apps.skin import skin_page


from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Dr.",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
  )

pages = {
    "Home": home_page,
    "Heart Disease Prediction": heart_page,
    "Tubercolosis Detection": tb_page,
    "Skin Cancer Classification": skin_page,
}

# For Horizontal Menu Layout
selected_page = option_menu(
        menu_title = None,
        options = list(pages.keys()),
        icons=['house', 'heart', 'lungs', 'person', 'robot'],
        orientation="horizontal",
    )

if selected_page in pages:
    pages[selected_page]()
else:
    st.markdown("### Invalid Page Selected")
