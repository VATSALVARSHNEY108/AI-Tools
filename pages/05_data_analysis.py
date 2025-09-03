import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.gemini_client import generate_text, initialize_gemini_client
from utils.file_processors import process_csv_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Initialize Gemini client
initialize_gemini_client()

st.title("📊 Data Analysis Tools")
st.markdown("Analyze and visualize data with AI assistance")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_insights' not in st.session_state:
    st.session_state.data_insights = ""

# File upload
uploaded_file = st.file_uploader(
    "Upload a CSV file:",
    type=['csv'],
    help="Upload a CSV file for analysis"
)

# Process uploaded file
if uploaded_file is not None:
    df = process_csv_file(uploaded_file)
    if df is not None:
        st.session_state.df = df
        st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Sample data option
if st.session_state.df is None:
    if st.button("📊 Use Sample Sales Data"):
        # Create sample data
        np.random.seed(42)
        sample_data = {
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 100),
            'Sales': np.random.normal(1000, 200, 100).astype(int),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Customer_Satisfaction': np.random.uniform(3, 5, 100).round(1)
        }
        st.session_state.df = pd.DataFrame(sample_data)
        st.success("✅ Sample dataset loaded!")

# Main analysis interface
def calculate_prediction_intervals(df, target_col, feature_cols):
    pass


if st.session_state.df is not None:
    df = st.session_state.df

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📏 Rows", df.shape[0])
    with col2:
        st.metric("📊 Columns", df.shape[1])
    with col3:
        st.metric("🔢 Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
    with col4:
        st.metric("📝 Text Columns", df.select_dtypes(include=['object']).shape[1])

    # Tabs for different analysis types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📋 Overview", "📈 Visualizations", "🤖 AI Insights", "📊 Statistics", "🔍 Custom Analysis", "🧠 ML Predictions"])

    with tab1:
        st.header("Dataset Overview")

        # Display first few rows
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10))

        # Basic info
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Column Information")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(info_df)

        with col2:
            st.subheader("Basic Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("No numeric columns found for statistics")

    with tab2:
        st.header("Data Visualizations")

        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
        )

        if viz_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", df.columns)
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_col = st.selectbox("Y-axis:", numeric_cols)
                else:
                    st.error("No numeric columns available for Y-axis")
                    y_col = None

            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Line Chart":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            date_cols = df.select_dtypes(include=['datetime64']).columns

            col1, col2 = st.columns(2)
            with col1:
                if len(date_cols) > 0:
                    x_col = st.selectbox("X-axis (time):", list(date_cols) + list(df.columns))
                else:
                    x_col = st.selectbox("X-axis:", df.columns)
            with col2:
                if len(numeric_cols) > 0:
                    y_col = st.selectbox("Y-axis:", numeric_cols)
                else:
                    st.error("No numeric columns available")
                    y_col = None

            if y_col:
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_cols)
                with col3:
                    color_col = st.selectbox("Color by:", [None] + list(df.columns))

                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                 title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Need at least 2 numeric columns for scatter plot")

        elif viz_type == "Histogram":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_to_plot = st.selectbox("Column:", numeric_cols)
                bins = st.slider("Number of bins:", 10, 100, 30)

                fig = px.histogram(df, x=col_to_plot, nbins=bins,
                                   title=f"Distribution of {col_to_plot}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No numeric columns available for histogram")

        elif viz_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    y_col = st.selectbox("Y-axis:", numeric_cols)
                with col2:
                    x_col = st.selectbox("Group by:", [None] + list(df.columns))

                fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No numeric columns available for box plot")

        elif viz_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr(method='pearson')
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Need at least 2 numeric columns for correlation heatmap")

    with tab3:
        st.header("AI-Powered Insights")

        if st.button("🧠 Generate Data Insights", type="primary"):
            with st.spinner("Analyzing data..."):
                # Prepare data summary for AI
                data_summary = f"""
                Dataset: {df.shape[0]} rows, {df.shape[1]} columns
                Columns: {', '.join(df.columns.tolist())}
                Numeric columns: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}

                Sample data (first 5 rows):
                {df.head().to_string()}

                Basic statistics:
                {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric data'}
                """

                prompt = f"""Analyze this dataset and provide insights:

                {data_summary}

                Please provide:
                1. Key observations about the data
                2. Notable patterns or trends
                3. Potential correlations
                4. Data quality observations
                5. Recommendations for further analysis
                """

                insights = generate_text(prompt, model="gemini-2.5-pro")
                st.session_state.data_insights = insights
                st.markdown("### 🔍 AI Insights:")
                st.write(insights)

        if st.session_state.data_insights:
            st.markdown("### Previous Insights:")
            st.write(st.session_state.data_insights)

        # Custom AI questions
        st.markdown("---")
        st.subheader("Ask AI About Your Data")
        custom_question = st.text_input("What would you like to know about your data?")

        if st.button("Get AI Answer"):
            if custom_question:
                with st.spinner("Getting answer..."):
                    data_context = f"Dataset with {df.shape[0]} rows and columns: {', '.join(df.columns.tolist())}"
                    prompt = f"Based on this data context: {data_context}, answer: {custom_question}"
                    answer = generate_text(prompt)
                    st.write(answer)
            else:
                st.warning("Please enter a question about your data.")

    with tab4:
        st.header("Statistical Analysis")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Descriptive Statistics")
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=list(numeric_cols)[:3])
                if selected_cols:
                    st.dataframe(df[selected_cols].describe())

            with col2:
                st.subheader("Missing Value Analysis")
                missing_data = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0])

            # Additional statistics
            if len(numeric_cols) >= 2:
                st.subheader("Correlation Analysis")
                corr_col1, corr_col2 = st.columns(2)
                with corr_col1:
                    col1_sel = st.selectbox("Column 1:", numeric_cols, key="corr1")
                with corr_col2:
                    col2_sel = st.selectbox("Column 2:", numeric_cols, key="corr2")

                if col1_sel != col2_sel:
                    correlation = df[col1_sel].corr(df[col2_sel], method='pearson')
                    st.metric("Correlation Coefficient", f"{correlation:.3f}")
        else:
            st.info("No numeric columns found for statistical analysis")

    with tab5:
        st.header("Custom Analysis")

        analysis_request = st.text_area(
            "Describe the analysis you want to perform:",
            placeholder="Example: Find the top 5 products by sales, or analyze sales trends by region"
        )

        if st.button("Perform Analysis", type="primary"):
            if analysis_request:
                with st.spinner("Performing custom analysis..."):
                    # Create a summary of the data for context
                    data_summary = f"""
                    Dataset columns: {', '.join(df.columns.tolist())}
                    Shape: {df.shape[0]} rows, {df.shape[1]} columns
                    Sample data: {df.head(3).to_string()}
                    """

                    prompt = f"""Based on this dataset:
                    {data_summary}

                    Perform this analysis: {analysis_request}

                    Provide specific insights, calculations, and recommendations based on the data structure shown."""

                    result = generate_text(prompt, model="gemini-2.5-pro")
                    st.markdown("### Analysis Results:")
                    st.write(result)
            else:
                st.warning("Please describe the analysis you want to perform.")

    with tab6:
        st.header("Machine Learning Predictions")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(numeric_cols) >= 2:
            st.subheader("🎯 Predictive Modeling")

            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("Target Variable (to predict):", numeric_cols)
                model_type = st.selectbox("Model Type:", ["Regression", "Classification", "Clustering"])

            with col2:
                feature_cols = st.multiselect("Feature Variables:",
                                              [col for col in df.columns if col != target_col],
                                              default=[col for col in numeric_cols if col != target_col][:3])

            if model_type == "Regression" and target_col and feature_cols:
                if st.button("🚀 Train Regression Model", type="primary"):
                    train_regression_model(df, target_col, feature_cols)

            elif model_type == "Classification" and target_col and feature_cols:
                if st.button("🚀 Train Classification Model", type="primary"):
                    train_classification_model(df, target_col, feature_cols)

            elif model_type == "Clustering":
                if st.button("🚀 Perform Clustering", type="primary"):
                    perform_clustering(df, feature_cols)

            # Advanced ML features
            st.markdown("---")
            st.subheader("🔬 Advanced ML Analysis")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 Feature Importance Analysis"):
                    if len(feature_cols) >= 2 and target_col:
                        analyze_feature_importance(df, target_col, feature_cols)

            with col2:
                if st.button("🎯 Prediction Intervals"):
                    if len(feature_cols) >= 2 and target_col:
                        calculate_prediction_intervals(df, target_col, feature_cols)

        else:
            st.info("Need at least 2 numeric columns for machine learning analysis")


def train_regression_model(df, target_col, feature_cols):
    """Train and evaluate regression models"""
    try:
        # Prepare data
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()

        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 10:
            st.warning("Need at least 10 samples for reliable modeling")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        results = {}

        with st.spinner("Training regression models..."):
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                results[name] = {"model": model, "mse": mse, "predictions": y_pred}

        # Display results
        st.subheader("🎯 Regression Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Performance:**")
            for name, result in results.items():
                st.metric(f"{name} MSE", f"{result['mse']:.4f}")

        with col2:
            # Prediction vs Actual plot
            best_model = min(results.items(), key=lambda x: x[1]['mse'])
            fig = px.scatter(x=y_test, y=best_model[1]['predictions'],
                             title=f"Predictions vs Actual ({best_model[0]})",
                             labels={'x': 'Actual', 'y': 'Predicted'})
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                          x1=y_test.max(), y1=y_test.max(), line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

        # AI interpretation
        interpretation_prompt = f"""
        Interpret these regression results:
        - Target variable: {target_col}
        - Features: {', '.join(feature_cols)}
        - Best model: {best_model[0]} with MSE: {best_model[1]['mse']:.4f}
        - Sample size: {len(X)}

        Provide insights about:
        1. Model performance quality
        2. Which features are likely most important
        3. Recommendations for improvement
        4. Business implications
        """

        interpretation = generate_text(interpretation_prompt, model="gemini-2.5-pro")
        st.markdown("### 🧠 AI Interpretation")
        st.write(interpretation)

    except Exception as e:
        st.error(f"Error in regression modeling: {e}")


def train_classification_model(df, target_col, feature_cols):
    """Train and evaluate classification models"""
    try:
        # Prepare data
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()

        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Encode target if it's categorical
        le = LabelEncoder()
        if y.dtype == 'object':
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y

        if len(np.unique(y_encoded)) < 2:
            st.warning("Target variable needs at least 2 different values for classification")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Train models
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = {}

        with st.spinner("Training classification models..."):
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {"model": model, "accuracy": accuracy, "predictions": y_pred}

        # Display results
        st.subheader("🎯 Classification Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Performance:**")
            for name, result in results.items():
                st.metric(f"{name} Accuracy", f"{result['accuracy']:.3f}")

        with col2:
            # Confusion matrix visualization
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, best_model[1]['predictions'])
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                            title=f"Confusion Matrix ({best_model[0]})")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in classification modeling: {e}")


def perform_clustering(df, feature_cols):
    """Perform clustering analysis"""
    try:
        if len(feature_cols) < 2:
            st.warning("Need at least 2 features for clustering")
            return

        # Prepare data
        X = df[feature_cols].dropna()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of clusters
        inertias = []
        K_range = range(2, min(11, len(X) // 2))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Elbow plot
        fig = px.line(x=list(K_range), y=inertias, title="Elbow Method for Optimal Clusters",
                      labels={'x': 'Number of Clusters', 'y': 'Inertia'})
        st.plotly_chart(fig, use_container_width=True)

        # Perform clustering with optimal k
        optimal_k = st.slider("Select number of clusters:", 2, min(10, len(X) // 2), 3)

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Add clusters to dataframe
        df_clustered = X.copy()
        df_clustered['Cluster'] = clusters

        # Visualize clusters
        if len(feature_cols) >= 2:
            fig = px.scatter(df_clustered, x=feature_cols[0], y=feature_cols[1],
                             color='Cluster', title="Clustering Results")
            st.plotly_chart(fig, use_container_width=True)

        # Cluster analysis
        st.subheader("📊 Cluster Analysis")
        cluster_summary = df_clustered.groupby('Cluster')[feature_cols].mean()
        st.dataframe(cluster_summary)

    except Exception as e:
        st.error(f"Error in clustering analysis: {e}")


def analyze_feature_importance(df, target_col, feature_cols):
    """Analyze feature importance using Random Forest"""
    try:
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()

        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Visualize
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance Analysis")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(importance_df)

    except Exception as e:
        st.error(f"Error in feature importance analysis: {e}")


def calculate_prediction_intervals(df, target_col, feature_cols):
    """Calculate prediction intervals for uncertainty quantification"""
    try:
        X = df[feature_cols].dropna()
        y = df[target_col].dropna()

        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Use multiple models for ensemble prediction
        from sklearn.ensemble import RandomForestRegressor

        models = []
        predictions = []

        # Bootstrap sampling for uncertainty estimation
        for i in range(10):
            # Random sample with replacement
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]

            model = RandomForestRegressor(n_estimators=50, random_state=i)
            model.fit(X_sample, y_sample)
            pred = model.predict(X)
            predictions.append(pred)

        # Calculate prediction intervals
        predictions_array = np.array(predictions)
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)

        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred

        # Create visualization
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': mean_pred,
            'Lower_95': lower_bound,
            'Upper_95': upper_bound
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'],
                                 mode='markers', name='Actual', marker_color='blue'))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted'],
                                 mode='lines', name='Predicted', line_color='red'))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Upper_95'],
                                 mode='lines', line_color='gray', showlegend=False))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Lower_95'],
                                 mode='lines', fill='tonexty', line_color='gray',
                                 name='95% Confidence Interval'))

        fig.update_layout(title="Predictions with Uncertainty Intervals")
        st.plotly_chart(fig, use_container_width=True)

        # Coverage analysis
        coverage = np.mean((y >= lower_bound) & (y <= upper_bound))
        st.metric("Prediction Interval Coverage", f"{coverage:.1%}")

    except Exception as e:
        st.error(f"Error in prediction interval calculation: {e}")

else:
st.info("👆 Upload a CSV file or use sample data to begin analysis")

# Data analysis capabilities
st.markdown("### 📊 Analysis Capabilities")
capabilities = [
    "📋 **Data Overview** - Basic dataset information and structure",
    "📈 **Visualizations** - Interactive charts and graphs",
    "🤖 **AI Insights** - Automated pattern detection and analysis",
    "📊 **Statistics** - Descriptive statistics and correlation analysis",
    "🔍 **Custom Analysis** - AI-powered custom data exploration",
    "📉 **Trend Analysis** - Time series and trend identification",
    "🎯 **Segmentation** - Data grouping and categorization",
    "⚠️ **Quality Assessment** - Missing data and outlier detection"
]

for capability in capabilities:
    st.markdown(capability)
