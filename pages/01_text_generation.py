import streamlit as st
from utils.gemini_client import generate_text, initialize_gemini_client

# Initialize Gemini client
initialize_gemini_client()

st.title("📝 Text Generation Tools")
st.markdown("Generate various types of content using AI")

# Text generation categories
tab1, tab2, tab3, tab4 = st.tabs(["✍️ Creative Writing", "📋 Summaries", "📧 Professional", "🧠 Brainstorming"])

with tab1:
    st.header("Creative Writing")

    writing_type = st.selectbox(
        "Choose writing type:",
        ["Story", "Poem", "Blog Post", "Product Description", "Social Media Post"]
    )

    topic = st.text_input("Enter topic or theme:")
    tone = st.selectbox("Select tone:", ["Casual", "Professional", "Humorous", "Serious", "Creative"])
    length = st.selectbox("Content length:", ["Short", "Medium", "Long"])

    if st.button("Generate Creative Content", type="primary"):
        if topic:
            with st.spinner("Generating content..."):
                prompt = f"Write a {length.lower()} {writing_type.lower()} about '{topic}' in a {tone.lower()} tone."
                result = generate_text(prompt)
                st.markdown("### Generated Content:")
                st.write(result)
        else:
            st.warning("Please enter a topic first.")

with tab2:
    st.header("Text Summarization")

    text_to_summarize = st.text_area("Enter text to summarize:", height=200)
    summary_length = st.selectbox("Summary length:", ["Brief", "Moderate", "Detailed"])

    if st.button("Generate Summary", type="primary"):
        if text_to_summarize:
            with st.spinner("Creating summary..."):
                prompt = f"Create a {summary_length.lower()} summary of the following text:\n\n{text_to_summarize}"
                result = generate_text(prompt)
                st.markdown("### Summary:")
                st.write(result)
        else:
            st.warning("Please enter text to summarize.")

with tab3:
    st.header("Professional Content")

    content_type = st.selectbox(
        "Content type:",
        ["Email", "Letter", "Report", "Proposal", "Press Release", "Job Description"]
    )

    col1, col2 = st.columns(2)
    with col1:
        purpose = st.text_input("Purpose/Goal:")
    with col2:
        audience = st.text_input("Target audience:")

    key_points = st.text_area("Key points to include:")

    if st.button("Generate Professional Content", type="primary"):
        if purpose and key_points:
            with st.spinner("Generating professional content..."):
                prompt = f"Write a professional {content_type.lower()} with the purpose: {purpose}. Target audience: {audience}. Include these key points: {key_points}"
                result = generate_text(prompt)
                st.markdown("### Generated Content:")
                st.write(result)
        else:
            st.warning("Please enter purpose and key points.")

with tab4:
    st.header("Brainstorming Assistant")

    brainstorm_type = st.selectbox(
        "Brainstorming type:",
        ["Ideas", "Solutions", "Questions", "Alternatives", "Improvements"]
    )

    challenge = st.text_area("Describe your challenge or topic:")
    constraints = st.text_input("Any constraints or requirements:")

    if st.button("Generate Ideas", type="primary"):
        if challenge:
            with st.spinner("Brainstorming..."):
                prompt = f"Generate creative {brainstorm_type.lower()} for this challenge: {challenge}. Constraints: {constraints}"
                result = generate_text(prompt)
                st.markdown("### Brainstorming Results:")
                st.write(result)
        else:
            st.warning("Please describe your challenge first.")

# Additional options
st.markdown("---")
with st.expander("⚙️ Advanced Options"):
    model_choice = st.selectbox("AI Model:", ["gemini-2.5-flash", "gemini-2.5-pro"])
    temperature = st.slider("Creativity Level:", 0.1, 1.0, 0.7, 0.1)

    st.info("Higher creativity levels produce more varied and creative outputs.")
