import streamlit as st # type: ignore
import joblib # type: ignore
import pandas as pd # type: ignore
import lime.lime_tabular # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import matplotlib.colors # type: ignore

# Initialize session state variables if they don't exist
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'prediction_probability' not in st.session_state:
    st.session_state['prediction_probability'] = None
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {
        'Pregnancies': 0.0,
        'Glucose': 0.0,
        'BloodPressure': 0.0,
        'SkinThickness': 0.0,
        'Insulin': 0.0,
        'BMI': 0.0,
        'DiabetesPedigreeFunction': 0.0,
        'Age': 0.0
    }
if 'prediction_section' not in st.session_state:
    st.session_state['prediction_section'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

# Load the scaled models and unscaled training/testing data
try:
    rf_model = joblib.load('rf_model_scaled.joblib')
    xgb_model = joblib.load('xgb_model_scaled.joblib')
    lgbm_model = joblib.load('lgbm_model_scaled.joblib')
    X_train = joblib.load('train_data.joblib')
    X_test = joblib.load('test_data.joblib')
    y_test = joblib.load('test_label.joblib')
    y_train = joblib.load('train_label.joblib')
    scaler = joblib.load('scaler_file.joblib')  # Load the fitted scaler
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Please ensure all necessary files are in the correct directory.")
    st.stop()

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
class_names = ['No Diabetes', 'Diabetes']

def map_lime_feature(lime_name):
    if "Glucose" in lime_name:
        return "Glucose"
    elif "Insulin" in lime_name:
        return "Insulin"
    elif "BMI" in lime_name:
        return "BMI"
    elif "Age" in lime_name:
        return "Age"
    elif "BloodPressure" in lime_name:
        return "BloodPressure"
    elif "DiabetesPedigreeFunction" in lime_name:
        return "DiabetesPedigreeFunction"
    elif "Pregnancies" in lime_name:
        return "Pregnancies"
    elif "SkinThickness" in lime_name:
        return "SkinThickness"
    return lime_name

def create_doctor_explanation(explanation_list, input_data_dict, predicted_class_name, feature_names):
    total_weight_diabetes = 0
    total_weight_no_diabetes = 0
    input_data_values_list = list(input_data_dict.values())
    feature_values_map = {feature_names[i]: input_data_values_list[i] for i in range(len(feature_names))}

    positive_contributions = sorted([(map_lime_feature(f), w) for f, w in explanation_list if w > 0], key=lambda x: x[1], reverse=True)
    negative_contributions = sorted([(map_lime_feature(f), abs(w)) for f, w in explanation_list if w < 0], key=lambda x: x[1], reverse=True)

    feature_contributions = {} 

    for feature, weight in explanation_list:
        if weight > 0:
            total_weight_diabetes += weight
        else:
            total_weight_no_diabetes += abs(weight)

    explanation_string = f"The AI model predicted '{predicted_class_name}' for this patient.\n\n"
    explanation_string += f"The total LIME weight of features contributing towards 'Diabetes' is {total_weight_diabetes * 100:.2f}%.\n"
    explanation_string += f"The total LIME weight of features contributing towards 'No Diabetes' is {total_weight_no_diabetes * 100:.2f}%.\n\n"

    if predicted_class_name == 'No Diabetes':
        if total_weight_diabetes > total_weight_no_diabetes:
            explanation_string += f"Even though the overall 'pull' from factors increasing the likelihood of 'Diabetes' ({total_weight_diabetes * 100:.2f}%) was stronger than the 'pull' from factors decreasing it ({total_weight_no_diabetes * 100:.2f}%), the model predicted 'No Diabetes'.\n"
            if negative_contributions:
                explanation_string += f"Think of it as a tug of war. While features like {', '.join([f'{f} with a significant negative pull of {w*100:.2f}%' for f, w in negative_contributions[:2]])} were strongly pulling towards 'No Diabetes', the collective influence of other factors pushing towards 'Diabetes' was greater in total weight.\n"
            explanation_string += "However, the specific values and combinations of the features that decrease the risk might have been critical enough in this case to lead the model to predict 'No Diabetes'.\n"
        else:
            explanation_string += f"In this scenario, the factors that decrease the likelihood of 'Diabetes' had a greater overall influence ({total_weight_no_diabetes * 100:.2f}%) than those that increase it ({total_weight_diabetes * 100:.2f}%).\n"
            if negative_contributions:
                explanation_string += f"Features like {', '.join([f'{f} with a negative pull of {w*100:.2f}%' for f, w in negative_contributions[:2]])} significantly contributed to this outcome.\n"
            explanation_string += "The balance of evidence favored a 'No Diabetes' prediction.\n"
    else:  # Diabetes
        if total_weight_no_diabetes > total_weight_diabetes:
            explanation_string += f"Interestingly, even though the total 'pull' from factors decreasing the likelihood of 'Diabetes' ({total_weight_no_diabetes * 100:.2f}%) was stronger than the 'pull' from factors increasing it ({total_weight_diabetes * 100:.2f}%), the model predicted 'Diabetes'.\n"
            if positive_contributions:
                explanation_string += f"Think of it as a tug of war. While features like {', '.join([f'{f} with a significant positive pull of {w*100:.2f}%' for f, w in positive_contributions[:2]])} were strongly pulling towards 'Diabetes', the collective influence of other factors decreasing the risk was greater in total weight.\n"
        else:
            explanation_string += f"In this case, the factors that increase the likelihood of 'Diabetes' had a greater overall influence ({total_weight_diabetes * 100:.2f}%) than those that decrease it ({total_weight_no_diabetes * 100:.2f}%).\n"
            if positive_contributions:
                explanation_string += f"Features like {', '.join([f'{f} with a positive pull of {w*100:.2f}%' for f, w in positive_contributions[:2]])} significantly contributed to this outcome.\n"

    explanation_string += "\nPlease remember this is an explanation of the AI's reasoning and should be considered alongside clinical judgment."
    return explanation_string


def get_personalized_recommendations(explanation_list, input_data_dict):
    recommendations = []
    positive_contributors = sorted([(map_lime_feature(f), w) for f, w in explanation_list if w > 0], key=lambda x: x[1], reverse=True)

    if st.session_state['prediction_result'] and "high risk" in st.session_state['prediction_result']:
        recommendations.append("Based on the prediction of high diabetes risk:")
        for feature, weight in positive_contributors:
            patient_value = input_data_dict.get(feature, 'N/A')
            if "Glucose" == feature:
                recommendations.append(f"- **Glucose:** Elevated level ({patient_value:.2f}) is a major contributing factor. Recommend immediate blood glucose monitoring and dietary changes.")
            elif "BMI" == feature:
                recommendations.append(f"- **BMI:** High BMI ({patient_value:.2f}) significantly increases risk. Recommend a comprehensive weight management program.")
            elif "Insulin" == feature:
                recommendations.append(f"- **Insulin:** High insulin level ({patient_value:.2f}) might indicate insulin resistance.")
            elif "Age" == feature and isinstance(patient_value, (int, float)) and patient_value >= 45:
                recommendations.append(f"- **Age:** Being over 45 ({int(patient_value)}) increases risk.")
    else:
        recommendations.append("Even though the predicted risk of diabetes is low, consider these contributing factors:")
        for feature, weight in positive_contributors:
            patient_value = input_data_dict.get(feature, 'N/A')
            if "Glucose" == feature and isinstance(patient_value, (int, float)) and patient_value >= 100:
                recommendations.append(f"- **Glucose:** Borderline high glucose level ({patient_value:.2f}) is noted.")
            elif "BMI" == feature and isinstance(patient_value, (int, float)) and patient_value >= 25:
                recommendations.append(f"- **BMI:** Overweight/Obese status (BMI: {patient_value:.2f}) increases future risk.")

    recommendations.append("\nContinue to encourage a healthy lifestyle, including balanced diet and regular physical activity.")
    return recommendations


def app():
    st.title("DIABETES RISK PREDICTION")
    st.write("Enter medical information to predict the likelihood of diabetes.")

    col1, col2 = st.columns(2)
    with col1:
        input_data = {}
        input_data[feature_names[0]] = st.number_input(f"{feature_names[0]}", value=st.session_state['input_data'].get(feature_names[0], 0.0))
        input_data[feature_names[1]] = st.number_input(f"{feature_names[1]}", value=st.session_state['input_data'].get(feature_names[1], 0.0))
        input_data[feature_names[2]] = st.number_input(f"{feature_names[2]}", value=st.session_state['input_data'].get(feature_names[2], 0.0))
        input_data[feature_names[3]] = st.number_input(f"{feature_names[3]}", value=st.session_state['input_data'].get(feature_names[3], 0.0))
    with col2:
        input_data[feature_names[4]] = st.number_input(f"{feature_names[4]}", value=st.session_state['input_data'].get(feature_names[4], 0.0))
        input_data[feature_names[5]] = st.number_input(f"{feature_names[5]}", value=st.session_state['input_data'].get(feature_names[5], 0.0))
        input_data[feature_names[6]] = st.number_input(f"{feature_names[6]}", value=st.session_state['input_data'].get(feature_names[6], 0.0))
        input_data[feature_names[7]] = st.number_input(f"{feature_names[7]}", value=st.session_state['input_data'].get(feature_names[7], 0.0))
        
    st.session_state['input_data'] = input_data

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([list(st.session_state['input_data'].values())], columns=feature_names)
            scaled_input = scaler.transform(input_df)
            scaled_input_df = pd.DataFrame(scaled_input, columns=input_df.columns)
            prediction_proba = lgbm_model.predict_proba(scaled_input_df)[0]
            prediction = np.argmax(prediction_proba)
            st.session_state['prediction_probability'] = prediction_proba
            
            if prediction == 1:
                st.session_state['prediction_result'] = "Person has high risk of having diabetes!"
            else:
                st.session_state['prediction_result'] = "Person has low risk of having diabetes!"
            
            st.session_state['prediction_section'] = 'results'
            
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

    if st.session_state['prediction_result']:
        st.subheader("Prediction:")
        if "high risk" in st.session_state['prediction_result']:
            st.error(st.session_state['prediction_result'])
        else:
            st.success(st.session_state['prediction_result'])
            
        if st.button("Explain Prediction & Get Recommendations"):
            st.session_state['prediction_section'] = 'explanation'
            st.rerun()

    if st.session_state['prediction_section'] == 'explanation':
        try:
            input_df_original = pd.DataFrame([list(st.session_state['input_data'].values())], columns=feature_names)
            scaled_input_original = scaler.transform(input_df_original).astype('float64')

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=scaler.transform(X_train).astype('float64'),
                feature_names=feature_names,
                class_names=class_names,
                mode='classification'
            )

            explanation = explainer.explain_instance(
                data_row=scaled_input_original[0],
                predict_fn=lgbm_model.predict_proba,
                num_features=len(st.session_state['input_data'])
            )

            st.title("LIME EXPLANATION")

            with st.container(border=True):
                explanation_list_diabetes = explanation.as_list(label=1)
                sorted_explanation_diabetes = sorted(explanation_list_diabetes, key=lambda x: x[1], reverse=True)
                features = [f for f, w in sorted_explanation_diabetes]
                weights = [w for f, w in sorted_explanation_diabetes]
                colors = ['green' if w < 0 else 'red' for w in weights]
                
                col1_exp, col2_exp = st.columns([1, 1])

                with col1_exp:
                    st.subheader("Explanation Plot")
                    if features:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.barh(features, weights, color=colors, edgecolor='black', linewidth=0.7)
                        ax.set_xlabel("LIME Weight", fontsize=8)
                        ax.set_ylabel("Feature", fontsize=8)
                        ax.set_title("Local Feature Importance", fontsize=10)
                        ax.tick_params(axis='both', which='major', labelsize=6)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)

                with col2_exp:
                    st.subheader("Feature Contributions")
                    input_data_list = []
                    for feature, weight in sorted_explanation_diabetes:
                        original_feature = map_lime_feature(feature)
                        original_value = st.session_state['input_data'].get(original_feature, 'N/A')
                        weight_percent = f"{weight * 100:.2f}%"
                        color = "green" if weight < 0 else "red"
                        input_data_list.append({
                            'Feature': original_feature,
                            'User Value': f"{original_value:.2f}" if isinstance(original_value, (int, float)) else original_value,
                            'LIME Weight (%)': f'<span style="color:{color};">{weight_percent}</span>'
                        })
                    
                    if input_data_list:
                        input_df_display = pd.DataFrame(input_data_list)
                        st.markdown(input_df_display.to_html(index=False, escape=False, justify='left'), unsafe_allow_html=True)

            explanation_list = explanation.as_list(label=1)
            predicted_class_index = np.argmax(lgbm_model.predict_proba(scaled_input_original))
            predicted_class_name = class_names[predicted_class_index]

            doctor_explanation = create_doctor_explanation(
                explanation_list=explanation_list,
                input_data_dict=st.session_state['input_data'],
                predicted_class_name=predicted_class_name,
                feature_names=feature_names
            )

            st.subheader("Reason for the Prediction:")
            st.info(doctor_explanation)    

            # Personalized Recommendations
            with st.container(border=True):
                recommendations = get_personalized_recommendations(explanation.as_list(label=1), st.session_state['input_data'])
                st.subheader("Personalized Recommendations Based on Contributing Factors:")
                for recommendation in recommendations:
                    st.info(recommendation)

                st.write("**Disclaimer:** This analysis provides insights but should not replace professional medical judgment.")

            if st.button("Go to Home"):
                st.session_state['prediction_section'] = None
                st.session_state['prediction_result'] = None
                st.session_state['current_page'] = 'home'
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred during explanation: {e}")

if __name__ == '__main__':
    app()