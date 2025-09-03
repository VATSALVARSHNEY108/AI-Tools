import streamlit as st
from utils.gemini_client import initialize_gemini_client

# Configure the page
st.set_page_config(
    page_title="AI Tools Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini client
initialize_gemini_client()

# Main page content
st.title("🤖 AI Tools Hub")
st.markdown("### Powered by Google Gemini")

st.markdown("""
Welcome to the comprehensive AI Tools Hub! This application provides access to various AI-powered tools 
and capabilities using Google's Gemini model.

**Available Tools:**
- 📝 **Text Generation** - Create content, summaries, and creative writing
- 👁️ **Image Analysis** - Analyze and describe images using AI vision
- 💻 **Code Assistant** - Generate, debug, and explain code
- 📄 **Document Processing** - Process and analyze documents (PDF, Word, Text)
- 📊 **Data Analysis** - Analyze CSV data and create visualizations
- 💬 **Chatbot** - Interactive conversation with AI
- 🔍 **Research Tools** - Research assistance and knowledge extraction

Use the sidebar navigation to explore different AI tool categories.
""")

# API Key status
if st.session_state.get('gemini_client'):
    st.success("✅ Gemini API is connected and ready!")
else:
    st.error("❌ Gemini API not configured. Please check your GEMINI_API_KEY environment variable.")

# Quick stats or features
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🧠 AI Model", "Gemini 2.5", "Latest")

with col2:
    st.metric("🛠️ Tool Categories", "7", "Categories")

with col3:
    st.metric("📱 Interface", "Streamlit", "Web App")

# Instructions
st.markdown("---")
st.markdown("### Getting Started")
st.markdown("""
1. Select a tool category from the sidebar
2. Follow the instructions on each page
3. Upload files or enter text as needed
4. Get AI-powered results instantly!
""")
