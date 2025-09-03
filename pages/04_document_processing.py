import streamlit as st
from utils.gemini_client import generate_text, chat_with_context, initialize_gemini_client
from utils.file_processors import process_text_file, process_pdf_file, process_docx_file, get_file_type

# Initialize Gemini client
initialize_gemini_client()

st.title("📄 Document Processing")
st.markdown("Process, analyze, and extract insights from documents")

# File upload
uploaded_file = st.file_uploader(
    "Upload a document:",
    type=['txt', 'pdf', 'docx'],
    help="Supported formats: TXT, PDF, DOCX"
)

# Initialize session state for document content
if 'document_content' not in st.session_state:
    st.session_state.document_content = ""
if 'document_name' not in st.session_state:
    st.session_state.document_name = ""

# Process uploaded file
if uploaded_file is not None:
    file_type = get_file_type(uploaded_file.name)

    with st.spinner("Processing document..."):
        if file_type == 'text':
            content = process_text_file(uploaded_file)
        elif file_type == 'pdf':
            content = process_pdf_file(uploaded_file)
        elif file_type == 'docx':
            content = process_docx_file(uploaded_file)
        else:
            st.error("Unsupported file format")
            content = None

    if content:
        st.session_state.document_content = content
        st.session_state.document_name = uploaded_file.name

        # Display document info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document", uploaded_file.name)
        with col2:
            st.metric("📏 Characters", len(content))
        with col3:
            st.metric("📝 Words", len(content.split()))

# Document processing tabs
if st.session_state.document_content:
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary", "🔍 Analysis", "❓ Q&A", "🏷️ Extract", "✨ Transform"])

    with tab1:
        st.header("Document Summary")

        col1, col2 = st.columns(2)
        with col1:
            summary_type = st.selectbox(
                "Summary type:",
                ["Executive Summary", "Key Points", "Abstract", "Brief Overview", "Detailed Summary"]
            )
        with col2:
            summary_length = st.selectbox(
                "Length:",
                ["Short", "Medium", "Long"]
            )

        if st.button("Generate Summary", type="primary"):
            with st.spinner("Creating summary..."):
                prompt = f"Create a {summary_length.lower()} {summary_type.lower()} of this document:\n\n{st.session_state.document_content}"
                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Summary:")
                st.write(result)

        # Preview document content
        with st.expander("📖 Document Preview"):
            preview_length = min(1000, len(st.session_state.document_content))
            st.text_area(
                "Document content (first 1000 characters):",
                st.session_state.document_content[:preview_length],
                height=200,
                disabled=True
            )

    with tab2:
        st.header("Document Analysis")

        analysis_type = st.selectbox(
            "Analysis type:",
            ["Sentiment Analysis", "Theme Analysis", "Writing Style", "Content Structure", "Readability",
             "Topic Modeling"]
        )

        if st.button("Analyze Document", type="primary"):
            with st.spinner("Analyzing document..."):
                if analysis_type == "Sentiment Analysis":
                    prompt = f"Analyze the sentiment and emotional tone of this document:\n\n{st.session_state.document_content}"
                elif analysis_type == "Theme Analysis":
                    prompt = f"Identify and analyze the main themes and topics in this document:\n\n{st.session_state.document_content}"
                elif analysis_type == "Writing Style":
                    prompt = f"Analyze the writing style, tone, and literary techniques used in this document:\n\n{st.session_state.document_content}"
                elif analysis_type == "Content Structure":
                    prompt = f"Analyze the structure, organization, and flow of this document:\n\n{st.session_state.document_content}"
                elif analysis_type == "Readability":
                    prompt = f"Assess the readability, complexity, and accessibility of this document:\n\n{st.session_state.document_content}"
                else:  # Topic Modeling
                    prompt = f"Identify and categorize the main topics discussed in this document:\n\n{st.session_state.document_content}"

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown(f"### {analysis_type} Results:")
                st.write(result)

    with tab3:
        st.header("Document Q&A")
        st.markdown("Ask questions about the document content")

        # Q&A interface
        question = st.text_input("Ask a question about the document:")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Get Answer", type="primary"):
                if question:
                    with st.spinner("Finding answer..."):
                        result = chat_with_context(question, st.session_state.document_content)
                        st.markdown("### Answer:")
                        st.write(result)
                else:
                    st.warning("Please enter a question.")

        with col2:
            if st.button("Clear Q&A"):
                st.rerun()

        # Sample questions
        st.markdown("#### 💡 Sample Questions:")
        sample_questions = [
            "What are the main points discussed?",
            "Who are the key people mentioned?",
            "What are the conclusions or recommendations?",
            "What dates or timeline information is provided?",
            "Are there any numbers or statistics mentioned?"
        ]

        for i, sample_q in enumerate(sample_questions):
            if st.button(f"📝 {sample_q}", key=f"sample_{i}"):
                with st.spinner("Finding answer..."):
                    result = chat_with_context(sample_q, st.session_state.document_content)
                    st.markdown("### Answer:")
                    st.write(result)

    with tab4:
        st.header("Information Extraction")

        extraction_type = st.multiselect(
            "Extract:",
            ["Names", "Dates", "Numbers", "Locations", "Organizations", "Email Addresses", "Phone Numbers", "Keywords"]
        )

        if st.button("Extract Information", type="primary"):
            if extraction_type:
                with st.spinner("Extracting information..."):
                    extraction_text = ", ".join(extraction_type)
                    prompt = f"Extract and list all {extraction_text.lower()} from this document:\n\n{st.session_state.document_content}"
                    result = generate_text(prompt, model="gemini-2.5-pro")
                    st.markdown("### Extracted Information:")
                    st.write(result)
            else:
                st.warning("Please select what to extract.")

    with tab5:
        st.header("Document Transformation")

        transform_type = st.selectbox(
            "Transform to:",
            ["Bullet Points", "FAQ Format", "Executive Brief", "Action Items", "Meeting Minutes", "Report Format"]
        )

        target_audience = st.selectbox(
            "Target audience:",
            ["General", "Technical", "Executive", "Academic", "Customer-facing"]
        )

        if st.button("Transform Document", type="primary"):
            with st.spinner("Transforming document..."):
                prompt = f"Transform this document into {transform_type.lower()} format for a {target_audience.lower()} audience:\n\n{st.session_state.document_content}"
                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown(f"### Transformed to {transform_type}:")
                st.write(result)

else:
    st.info("👆 Upload a document to begin processing")

    # Document processing capabilities
    st.markdown("### 📚 Processing Capabilities")
    capabilities = [
        "📋 **Summarization** - Create executive summaries, abstracts, and key points",
        "🔍 **Analysis** - Sentiment analysis, theme extraction, and content structure",
        "❓ **Q&A** - Interactive question-answering about document content",
        "🏷️ **Information Extraction** - Extract names, dates, locations, and key data",
        "✨ **Transformation** - Convert to different formats and styles",
        "📊 **Insights** - Generate insights and recommendations",
        "🎯 **Focus Areas** - Identify main topics and important sections",
        "📈 **Metrics** - Word count, readability scores, and content analysis"
    ]

    for capability in capabilities:
        st.markdown(capability)

    st.markdown("### 📁 Supported Formats")
    st.markdown("- **TXT** - Plain text files")
    st.markdown("- **PDF** - Portable Document Format")
    st.markdown("- **DOCX** - Microsoft Word documents")
