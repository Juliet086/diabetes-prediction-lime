# --- evaluation_page.py ---
import streamlit as st  # type: ignore
import joblib # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc # type: ignore



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


def app():
    st.title("Model Evaluation and Comparison")
    st.write("Evaluating and comparing the performance of different models used for diabetes prediction with tuned hyperparameters using RandomizedSearch with Standardization and Regularization.")
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    # Get the best estimators from RandomizedSearchCV
    best_rf_model = rf_model.best_estimator_
    best_xgb_model = xgb_model.best_estimator_
    best_lgbm_model = lgbm_model.best_estimator_

    models = {'Random Forest': best_rf_model, 'XGBoost': best_xgb_model, 'LightGBM': best_lgbm_model}
    model_names = list(models.keys())

    # Prepare dataframes to store metrics
    train_metrics_df = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    test_metrics_df = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'])

    


    for name, model_obj in models.items():
        # Predictions
        y_train_pred = model_obj.predict(X_train_scaled)
        y_test_pred = model_obj.predict(X_test_scaled)
        y_test_proba = model_obj.predict_proba(X_test_scaled)[:, 1] if hasattr(model_obj, 'predict_proba') else None

        # Training metrics
        train_metrics = {
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'Precision': precision_score(y_train, y_train_pred, zero_division=0),
            'Recall': recall_score(y_train, y_train_pred, zero_division=0),
            'F1-Score': f1_score(y_train, y_train_pred, zero_division=0)
        }
        train_metrics_df[name] = pd.Series(train_metrics)

        # Testing metrics
        test_metrics = {
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred, zero_division=0),
            'Recall': recall_score(y_test, y_test_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_test_pred, zero_division=0),
            'ROC AUC': roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None
        }
        test_metrics_df[name] = pd.Series(test_metrics)


    st.subheader("Model Performance Metrics")
    st.subheader("Training Data")
    st.write(train_metrics_df)
    st.subheader("Testing Data")
    st.write(test_metrics_df)

    col_rf, col_xgb, col_lgbm = st.columns(3)

    for (name, model_obj), col in zip(models.items(), [col_rf, col_xgb, col_lgbm]):
        with col:
            st.success(f"{name}")  # Use st.success() for the heading

            st.subheader("Confusion Matrix")
            st.markdown("<style>subheader { font-size: 1.0em; }</style>", unsafe_allow_html=True)

            try:
                y_pred = model_obj.predict(X_test_scaled)
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                            xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 10}, cbar=False)
                ax_cm.set_xlabel('Predicted', fontsize=0.9*16)
                ax_cm.set_ylabel('True', fontsize=0.9*16)
                ax_cm.set_title(f"Confusion Matrix for {name}", fontsize=1.1*16)
                st.pyplot(fig_cm)
            except Exception as e_pred:
                st.error(f"Error generating confusion matrix for {name}: {e_pred}")

            st.subheader(f"ROC AUC Curve", help="Receiver Operating Characteristic curve")
            st.markdown("<style>subheader { font-size: 1.0em; }</style>", unsafe_allow_html=True)
            try:
                if hasattr(model_obj, 'predict_proba'):
                    y_proba = model_obj.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                    ax_roc.set_xlabel('False Positive Rate', fontsize=0.9*16)
                    ax_roc.set_ylabel('True Positive Rate', fontsize=0.9*16)
                    ax_roc.set_title(f'ROC Curve for {name}', fontsize=1.1*16)
                    ax_roc.legend(loc="lower right", fontsize=0.9*16)
                    st.pyplot(fig_roc)
                else:
                    st.info(f"Probability predictions not available for {name} ROC AUC.")
            except Exception as e_roc:
                st.error(f"Error generating ROC AUC curve for {name}: {e_roc}")

            st.subheader(f"Precision-Recall Curve", help="Precision vs Recall curve")
            st.markdown("<style>subheader { font-size: 1.0em; }</style>", unsafe_allow_html=True)
            try:
                if hasattr(model_obj, 'predict_proba'):
                    y_proba = model_obj.predict_proba(X_test_scaled)[:, 1]
                    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                    pr_auc = auc(recall, precision)
                    fig_pr, ax_pr = plt.subplots()
                    ax_pr.plot(recall, precision, color='green', lw=2, label=f'AUC = {pr_auc:.2f}')
                    ax_pr.set_xlabel('Recall', fontsize=0.9*16)
                    ax_pr.set_ylabel('Precision', fontsize=0.9*16)
                    ax_pr.set_title(f'PR Curve for {name}', fontsize=1.1*16)
                    ax_pr.legend(loc="lower left", fontsize=0.9*16)
                    st.pyplot(fig_pr)
                else:
                    st.info(f"Probability predictions not available for {name} PR curve.")
            except Exception as e_pr:
                st.error(f"Error generating PR curve for {name}: {e_pr}")

            st.markdown("---")

    st.subheader("Feature Importance", help="Top features influencing the prediction")
    with st.container(border=True):
        col_rf_fi, col_xgb_fi, col_lgbm_fi = st.columns(3)

        def plot_feature_importance(model, model_name, col):
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names_list = list(feature_names) if feature_names is not None else [f"feature_{i+1}" for i in range(len(importance))]
                importance_df = pd.DataFrame({'Feature': feature_names_list, 'Importance': importance})
                importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

                with col:
                    st.markdown(model_name)
                    fig_importance, ax_importance = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax_importance,legend=False)
                    ax_importance.set_title("Top Feature Importance", fontsize=1.2*16)
                    ax_importance.set_xlabel("Importance", fontsize=1.0*16)
                    ax_importance.set_ylabel("Feature", fontsize=1.0*16)
                    ax_importance.tick_params(axis='both', which='major', labelsize=0.9*16)
                    st.pyplot(fig_importance)
            else:
                with col:
                    st.info(f"Feature importance not available for {model_name}")

        plot_feature_importance(models['Random Forest'], "Random Forest", col_rf_fi)
        plot_feature_importance(models['XGBoost'], "XGBoost", col_xgb_fi)
        plot_feature_importance(models['LightGBM'], "LightGBM", col_lgbm_fi)

    if st.button("Go to Home"):
        st.session_state['current_page'] = 'home'
        st.rerun()

if __name__ == '__main__':
    app()