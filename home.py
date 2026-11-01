import streamlit as st # type: ignore
from PIL import Image # type: ignore
 
def app():
    col1, col2 = st.columns([2, 2])

    with col1:
        try:
            image = Image.open('diaimg.jpg')
            st.image(image, caption="Diabetes Prediction App", width=600)
        except FileNotFoundError:
            st.info("Placeholder image not found.")

    with col2:
        st.title("DIABETES RISK PREDICTION APP WITH EXPLAINER")
        st.info("**WHY THIS APPLICATION?** This user-friendly tool is designed to assist healthcare professionals in predicting the risk of Type 2 diabetes in their patients. By integrating a LIME explainer, the app provides valuable insights into which factors are most influential in each prediction. This aims to foster more informed clinical judgment.")
        st.markdown("---")
        
        col2_row1, col2_row2, col2_row3 = st.columns([1, 1.2, 1])

        with col2_row1:
            if st.button("📈 PREDICTION", key="hp_prediction_button"):
                st.session_state['current_page'] = 'prediction'
                st.rerun()

        with col2_row2:
            if st.button("📊 MODEL EVALUATION", key="hp_evaluation_button"):
                st.session_state['current_page'] = 'evaluation'
                st.rerun()

        with col2_row3:
            if st.button("🔆 DATA INSIGHTS", key="hp_insights_button"):
                st.session_state['current_page'] = 'insights'
                st.rerun()

if __name__ == '__main__':
    app()