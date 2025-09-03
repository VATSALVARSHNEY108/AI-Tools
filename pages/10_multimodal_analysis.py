import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.gemini_client import generate_text, analyze_image_with_prompt, initialize_gemini_client
from utils.file_processors import process_image_file, process_csv_file, process_text_file, process_pdf_file
import json
import time

# Initialize Gemini client
initialize_gemini_client()

st.title("🔗 Multi-Modal AI Analysis")
st.markdown("Combine text, images, and data for comprehensive AI-powered insights")

# Initialize session state
if 'multimodal_data' not in st.session_state:
    st.session_state.multimodal_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

tab1, tab2, tab3, tab4 = st.tabs(["📥 Data Input", "🔍 Cross-Modal Analysis", "🧠 AI Synthesis", "📊 Insights Dashboard"])

with tab1:
    st.header("Multi-Modal Data Input")

    # Create columns for different input types
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📝 Text Data")

        text_input_type = st.radio("Text Input:", ["Direct Input", "File Upload"])

        if text_input_type == "Direct Input":
            text_data = st.text_area("Enter text data:", height=200,
                                     placeholder="Paste your text content here...")
            if text_data:
                st.session_state.multimodal_data['text'] = text_data

        else:
            text_file = st.file_uploader("Upload text file:", type=['txt', 'pdf', 'docx'], key="text_upload")
            if text_file:
                if text_file.type == "application/pdf":
                    content = process_pdf_file(text_file)
                elif text_file.name.endswith('.docx'):
                    content = process_docx_file(text_file)
                else:
                    content = process_text_file(text_file)

                if content:
                    st.session_state.multimodal_data['text'] = content
                    st.success("✅ Text data loaded")

    with col2:
        st.subheader("🖼️ Image Data")

        image_files = st.file_uploader("Upload images:", type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
                                       accept_multiple_files=True, key="image_upload")

        if image_files:
            st.session_state.multimodal_data['images'] = []
            for i, img_file in enumerate(image_files):
                image_bytes, image = process_image_file(img_file)
                if image_bytes and image:
                    st.session_state.multimodal_data['images'].append({
                        'name': img_file.name,
                        'bytes': image_bytes,
                        'image': image
                    })
                    st.image(image, caption=img_file.name, width=200)

            st.success(f"✅ {len(image_files)} images loaded")

    with col3:
        st.subheader("📊 Structured Data")

        data_input_type = st.radio("Data Input:", ["CSV Upload", "Manual JSON"])

        if data_input_type == "CSV Upload":
            csv_file = st.file_uploader("Upload CSV:", type=['csv'], key="csv_upload")
            if csv_file:
                df = process_csv_file(csv_file)
                if df is not None:
                    st.session_state.multimodal_data['dataframe'] = df
                    st.dataframe(df.head())
                    st.success("✅ CSV data loaded")

        else:
            json_data = st.text_area("Enter JSON data:", height=200,
                                     placeholder='{"key": "value", "data": [...]}')
            if json_data:
                try:
                    parsed_json = json.loads(json_data)
                    st.session_state.multimodal_data['json'] = parsed_json
                    st.json(parsed_json)
                    st.success("✅ JSON data loaded")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")

with tab2:
    st.header("Cross-Modal Analysis")

    if st.session_state.multimodal_data:
        available_data = list(st.session_state.multimodal_data.keys())
        st.info(f"Available data types: {', '.join(available_data)}")

        # Analysis type selection
        analysis_type = st.selectbox("Analysis Type:", [
            "Correlation Analysis", "Pattern Recognition", "Sentiment-Visual Alignment",
            "Data-Text Validation", "Comprehensive Comparison", "Trend Analysis"
        ])

        # Custom analysis query
        analysis_query = st.text_area("Analysis Focus:",
                                      placeholder="What specific insights are you looking for across your data types?")

        if st.button("🔍 Perform Cross-Modal Analysis", type="primary"):
            with st.spinner("Analyzing across all data modalities..."):
                perform_cross_modal_analysis(analysis_type, analysis_query)
    else:
        st.info("Please upload data in the Data Input tab first.")


def perform_cross_modal_analysis(analysis_type, query):
    """Perform analysis across multiple data modalities"""

    results = {}
    data = st.session_state.multimodal_data

    # Text analysis
    if 'text' in data:
        st.subheader("📝 Text Analysis")
        text_prompt = f"Analyze this text focusing on {analysis_type}: {data['text'][:2000]}"
        text_result = generate_text(text_prompt, model="gemini-2.5-pro")
        results['text_analysis'] = text_result
        st.write(text_result)

    # Image analysis
    if 'images' in data:
        st.subheader("🖼️ Image Analysis")
        image_results = []

        for img_data in data['images'][:3]:  # Analyze first 3 images
            img_prompt = f"Analyze this image for {analysis_type}. Focus on: {query}"
            img_result = analyze_image_with_prompt(img_data['bytes'], img_prompt)
            image_results.append({
                'name': img_data['name'],
                'analysis': img_result
            })

            with st.expander(f"Analysis: {img_data['name']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img_data['image'], width=200)
                with col2:
                    st.write(img_result)

        results['image_analysis'] = image_results

    # Data analysis
    if 'dataframe' in data:
        st.subheader("📊 Data Analysis")
        df = data['dataframe']

        # Statistical summary
        data_summary = f"Dataset shape: {df.shape}\nColumns: {', '.join(df.columns.tolist())}\n"
        if len(df.select_dtypes(include=['number']).columns) > 0:
            data_summary += f"Statistics:\n{df.describe().to_string()}"

        data_prompt = f"Analyze this dataset for {analysis_type}: {data_summary}"
        data_result = generate_text(data_prompt, model="gemini-2.5-pro")
        results['data_analysis'] = data_result
        st.write(data_result)

        # Create visualization
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            fig = px.scatter_matrix(df[numeric_cols.tolist()[:4]], title="Data Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

    # Store results for synthesis
    st.session_state.analysis_results[f"analysis_{int(time.time())}"] = {
        'type': analysis_type,
        'query': query,
        'results': results,
        'timestamp': time.time()
    }


with tab3:
    st.header("AI Synthesis & Integration")

    if st.session_state.analysis_results:
        latest_analysis = list(st.session_state.analysis_results.keys())[-1]
        analysis_data = st.session_state.analysis_results[latest_analysis]

        st.subheader("Synthesis Options")

        synthesis_type = st.selectbox("Synthesis Approach:", [
            "Unified Insights", "Contradiction Analysis", "Pattern Correlation",
            "Predictive Synthesis", "Decision Support", "Strategic Recommendations"
        ])

        synthesis_focus = st.text_area("Synthesis Focus:",
                                       placeholder="What specific questions should the synthesis address?")

        if st.button("🧠 Generate AI Synthesis", type="primary"):
            with st.spinner("Synthesizing insights across all modalities..."):
                generate_synthesis(analysis_data, synthesis_type, synthesis_focus)
    else:
        st.info("Perform cross-modal analysis first to enable synthesis.")


def generate_synthesis(analysis_data, synthesis_type, focus):
    """Generate comprehensive synthesis across all analyzed data"""

    # Prepare synthesis prompt
    synthesis_prompt = f"""
    As an expert analyst, synthesize insights from multiple data sources:

    Analysis Type: {analysis_data['type']}
    Synthesis Approach: {synthesis_type}
    Focus: {focus}

    Data Sources Analyzed:
    """

    results = analysis_data['results']

    if 'text_analysis' in results:
        synthesis_prompt += f"\nText Analysis: {results['text_analysis'][:1000]}"

    if 'image_analysis' in results:
        image_insights = [img['analysis'][:500] for img in results['image_analysis']]
        synthesis_prompt += f"\nImage Analysis: {'; '.join(image_insights)}"

    if 'data_analysis' in results:
        synthesis_prompt += f"\nData Analysis: {results['data_analysis'][:1000]}"

    synthesis_prompt += f"""

    Please provide:
    1. Unified insights that connect findings across all data types
    2. Key patterns and correlations discovered
    3. Contradictions or inconsistencies (if any)
    4. Actionable recommendations
    5. Areas for further investigation
    6. Confidence levels for major findings

    Focus on {synthesis_type.lower()} and address: {focus}
    """

    synthesis_result = generate_text(synthesis_prompt, model="gemini-2.5-pro")

    st.markdown("### 🎯 Comprehensive Synthesis")
    st.write(synthesis_result)

    # Create visual summary
    create_synthesis_visualization(results)


def create_synthesis_visualization(results):
    """Create visualizations to support synthesis"""

    st.markdown("### 📊 Synthesis Visualization")

    # Create a summary chart of findings
    categories = []
    scores = []

    if 'text_analysis' in results:
        categories.append("Text Insights")
        scores.append(len(results['text_analysis'].split()) / 10)  # Rough complexity score

    if 'image_analysis' in results:
        categories.append("Visual Insights")
        scores.append(len(results['image_analysis']))

    if 'data_analysis' in results:
        categories.append("Data Insights")
        scores.append(len(results['data_analysis'].split()) / 10)

    if categories:
        fig = go.Figure(data=[
            go.Bar(x=categories, y=scores,
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ])
        fig.update_layout(title="Analysis Depth by Modality",
                          yaxis_title="Insight Complexity Score")
        st.plotly_chart(fig, use_container_width=True)


with tab4:
    st.header("Insights Dashboard")

    if st.session_state.analysis_results:
        # Analysis history
        st.subheader("Analysis History")

        analysis_df = pd.DataFrame([
            {
                "ID": key,
                "Type": data["type"],
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"])),
                "Modalities": len(data["results"])
            }
            for key, data in st.session_state.analysis_results.items()
        ])

        st.dataframe(analysis_df, use_container_width=True)

        # Detailed view
        selected_analysis = st.selectbox("View Details:", list(st.session_state.analysis_results.keys()))

        if selected_analysis:
            analysis_detail = st.session_state.analysis_results[selected_analysis]

            st.subheader(f"Analysis: {analysis_detail['type']}")
            st.write(f"**Query:** {analysis_detail['query']}")

            # Results breakdown
            results = analysis_detail['results']

            col1, col2, col3 = st.columns(3)

            with col1:
                if 'text_analysis' in results:
                    with st.expander("📝 Text Results"):
                        st.write(results['text_analysis'])

            with col2:
                if 'image_analysis' in results:
                    with st.expander("🖼️ Image Results"):
                        for img in results['image_analysis']:
                            st.write(f"**{img['name']}:** {img['analysis'][:200]}...")

            with col3:
                if 'data_analysis' in results:
                    with st.expander("📊 Data Results"):
                        st.write(results['data_analysis'])

        # Export functionality
        st.subheader("Export & Sharing")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export All Results"):
                export_data = {
                    "multimodal_data_summary": {
                        "data_types": list(st.session_state.multimodal_data.keys()),
                        "analysis_count": len(st.session_state.analysis_results)
                    },
                    "analyses": st.session_state.analysis_results
                }

                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"multimodal_analysis_report_{int(time.time())}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("🔄 Clear All Data"):
                st.session_state.multimodal_data = {}
                st.session_state.analysis_results = {}
                st.success("All data cleared!")
                st.rerun()

    else:
        st.info("No analysis results yet. Start by uploading data and performing cross-modal analysis.")

# Advanced features section
st.markdown("---")
st.subheader("🚀 Advanced Multi-Modal Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔗 Data Fusion**")
    st.write("Combine insights from text, images, and structured data for unified understanding")

with col2:
    st.markdown("**🎯 Context Correlation**")
    st.write("Find hidden patterns and relationships across different data modalities")

with col3:
    st.markdown("**🧠 AI Synthesis**")
    st.write("Generate comprehensive insights that no single data type could provide alone")

# Tips for multi-modal analysis
with st.expander("💡 Multi-Modal Analysis Tips"):
    st.markdown("""
    **Maximizing Multi-Modal Insights:**

    **🎯 Best Practices:**
    - Upload complementary data types that relate to the same topic
    - Use specific analysis queries to guide the AI focus
    - Look for patterns that emerge across multiple modalities

    **🔍 Analysis Strategies:**
    - **Correlation Analysis**: Find relationships between visual, textual, and numerical data
    - **Validation**: Use one data type to validate insights from another
    - **Triangulation**: Confirm findings using multiple data sources

    **📊 Data Preparation:**
    - Ensure data quality across all modalities
    - Use consistent time periods or contexts
    - Consider the scale and granularity of different data types

    **🧠 Synthesis Techniques:**
    - Look for contradictions that reveal deeper insights
    - Identify emergent patterns not visible in individual analyses
    - Generate actionable recommendations based on combined insights
    """)