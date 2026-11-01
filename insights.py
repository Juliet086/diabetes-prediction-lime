# --- insights_page.py ---
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

def app():
    st.title("Dataset Insights")
    st.write("Explore the characteristics of the diabetes dataset.")

    # Load the diabetes dataset directly
    try:
        df_loaded = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        df_loaded = None
        st.warning("dataset not loaded")

    if df_loaded is not None:
        st.subheader("Dataset Overview")
        st.info(f"Number of rows: {df_loaded.shape[0]}")
        st.info(f"Number of columns: {df_loaded.shape[1]}")
        st.dataframe(df_loaded.head())

        st.subheader("Descriptive Statistics")
        st.dataframe(df_loaded.describe())

        st.subheader("Visualizations")

        # --- Outcome Distribution and Correlation Matrix Side by Side ---
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.success("Outcome Distribution")
                target_column = 'Outcome'  # Replace with your actual target column name
                if target_column in df_loaded.columns:
                    fig_target, ax_target = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=df_loaded, x=target_column, ax=ax_target)
                    ax_target.set_title(f"Distribution of '{target_column}'", fontsize=12)
                    ax_target.set_xlabel(target_column, fontsize=10)
                    ax_target.set_ylabel("Count", fontsize=10)
                    ax_target.tick_params(axis='both', which='major', labelsize=8)
                    st.pyplot(fig_target)
                else:
                    st.warning(f"Target column '{target_column}' not found.")

        with col2:
            with st.container(border=True):
                st.success("Correlation Heatmap")
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                corr_matrix = df_loaded.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 8}, ax=ax_corr)
                ax_corr.set_title("Correlation Matrix", fontsize=12)
                ax_corr.tick_params(axis='both', which='major', labelsize=8)
                st.pyplot(fig_corr)

        # --- Feature Distributions ---
        with st.container(border=True):
            st.subheader("Feature Distributions")
            st.info("The histogram with KDE shows the frequency of each value and provides a sense of the overall shape of the distribution for the entire dataset.")
            numerical_cols = df_loaded.select_dtypes(include=['number']).columns.tolist()
            if 'Outcome' in numerical_cols:
                numerical_cols.remove('Outcome')

            num_cols_to_plot = len(numerical_cols)
            cols = st.columns(3)  # Create 3 columns for feature distributions

            for i, col in enumerate(numerical_cols):
                with cols[i % 3]:
                    fig_hist, ax_hist = plt.subplots(figsize=(4, 3))
                    sns.histplot(data=df_loaded, x=col, kde=True, ax=ax_hist)
                    ax_hist.set_title(f"'{col}'", fontsize=10)
                    ax_hist.set_xlabel(col, fontsize=8)
                    ax_hist.set_ylabel("Frequency", fontsize=8)
                    ax_hist.tick_params(axis='both', which='major', labelsize=6)
                    st.pyplot(fig_hist)

        # --- Box Plots by Target ---
        with st.container(border=True):
            st.subheader("Box Plots by Target")
            st.info("The box plot gives you a concise summary of the distribution and facilitates comparison between groups.")

            numerical_cols_for_box = df_loaded.select_dtypes(include=['number']).columns.tolist()
            if 'Outcome' in numerical_cols_for_box:
                numerical_cols_for_box.remove('Outcome')

            num_cols_to_plot_box = len(numerical_cols_for_box)
            cols_box = st.columns(3)  # Create 3 columns for box plots

            for i, col in enumerate(numerical_cols_for_box):
                with cols_box[i % 3]:
                    fig_box, ax_box = plt.subplots(figsize=(4, 3))
                    sns.boxplot(data=df_loaded, x='Outcome', y=col, ax=ax_box)
                    ax_box.set_title(f"'{col}' by Outcome", fontsize=10)
                    ax_box.set_xlabel('Outcome', fontsize=8)
                    ax_box.set_ylabel(col, fontsize=8)
                    ax_box.tick_params(axis='both', which='major', labelsize=6)
                    st.pyplot(fig_box)

        # --- Add more insights and visualizations as needed ---

        if st.button("Go to Home"):
            st.session_state['current_page'] = 'home'
            st.rerun()

    else:
        st.info("Please upload the 'diabetes.csv' file to view dataset insights.")

if __name__ == '__main__':
    app()