# apps/home_app.py
import streamlit as st

def home_page():
    st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Medical Analysis Assistant</h1>
    </div>
    """,
    unsafe_allow_html=True,
    )
    
    st.markdown(
    """
    <div style='text-align: center; font-family: "Courier New", Courier, monospace;'>
Welcome to your comprehensive platform for medical predictions and analysis.    </div>
    """,unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("")
    st.markdown("")
   
    
    st.subheader('What can you do?')
    st.markdown('''
            - Heart Disease Prediction: Assess the likelihood of heart disease.\n
            - Tuberculosis Detection: Identify potential signs of tuberculosis.\n
            - Skin Cancer Classification: Evaluate skin conditions for possible cancer.\n
    '''
    )

    st.markdown("")
    st.markdown("")
   
    st.caption(
    "<span style='font-size: small;'>Note: This app is not a substitute for professional medical advice.</span>\n\n"
    "<span style='font-size: small;'>Always consult with a healthcare professional for accurate diagnosis and treatment.</span>",
    unsafe_allow_html=True
    )


if __name__ == "__main__":
    home_page()


