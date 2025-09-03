import streamlit as st
from utils.gemini_client import analyze_image_with_prompt, initialize_gemini_client
from utils.file_processors import process_image_file
from PIL import Image

# Initialize Gemini client
initialize_gemini_client()

st.title("👁️ Image Analysis Tools")
st.markdown("Analyze and describe images using AI vision capabilities")

# Image upload
uploaded_file = st.file_uploader(
    "Upload an image:",
    type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
    help="Supported formats: JPG, JPEG, PNG, GIF, BMP, WEBP"
)

if uploaded_file is not None:
    # Process and display image
    image_bytes, image = process_image_file(uploaded_file)

    if image_bytes and image:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.markdown("### Image Information")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")
            st.write(f"**File size:** {len(image_bytes)} bytes")

        # Analysis options
        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔍 General Analysis", "📝 Detailed Description", "🏷️ Object Detection", "🎨 Custom Analysis"])

        with tab1:
            st.header("General Image Analysis")
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    prompt = "Analyze this image and provide a comprehensive description including objects, people, setting, colors, mood, and any notable details."
                    result = analyze_image_with_prompt(image_bytes, prompt)
                    st.markdown("### Analysis Result:")
                    st.write(result)

        with tab2:
            st.header("Detailed Description")
            description_style = st.selectbox(
                "Description style:",
                ["Technical", "Creative", "Scientific", "Artistic", "Journalistic"]
            )

            if st.button("Generate Description", type="primary"):
                with st.spinner("Generating description..."):
                    prompt = f"Provide a detailed {description_style.lower()} description of this image, focusing on all visual elements, composition, and context."
                    result = analyze_image_with_prompt(image_bytes, prompt)
                    st.markdown("### Detailed Description:")
                    st.write(result)

        with tab3:
            st.header("Object and Scene Detection")
            detection_focus = st.multiselect(
                "Focus on:",
                ["People", "Objects", "Text", "Animals", "Vehicles", "Buildings", "Nature", "Food"],
                default=["People", "Objects"]
            )

            if st.button("Detect Objects", type="primary"):
                with st.spinner("Detecting objects..."):
                    focus_text = ", ".join(detection_focus)
                    prompt = f"Identify and list all {focus_text.lower()} visible in this image. Provide specific details about their location, appearance, and any notable characteristics."
                    result = analyze_image_with_prompt(image_bytes, prompt)
                    st.markdown("### Detection Results:")
                    st.write(result)

        with tab4:
            st.header("Custom Analysis")
            custom_prompt = st.text_area(
                "Enter your custom analysis request:",
                placeholder="Example: What emotions are conveyed in this image? / Is this image suitable for a professional website? / What story does this image tell?"
            )

            if st.button("Custom Analysis", type="primary"):
                if custom_prompt:
                    with st.spinner("Performing custom analysis..."):
                        result = analyze_image_with_prompt(image_bytes, custom_prompt)
                        st.markdown("### Custom Analysis Result:")
                        st.write(result)
                else:
                    st.warning("Please enter your analysis request.")

        # Additional features
        st.markdown("---")
        with st.expander("🛠️ Additional Features"):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Generate Alt Text"):
                    with st.spinner("Generating alt text..."):
                        prompt = "Generate concise, descriptive alt text for this image that would be suitable for web accessibility."
                        result = analyze_image_with_prompt(image_bytes, prompt)
                        st.markdown("**Alt Text:**")
                        st.code(result)

            with col2:
                if st.button("Safety Assessment"):
                    with st.spinner("Assessing image..."):
                        prompt = "Assess this image for content appropriateness, safety, and potential concerns for different audiences (general, children, professional)."
                        result = analyze_image_with_prompt(image_bytes, prompt)
                        st.markdown("**Safety Assessment:**")
                        st.write(result)

else:
    st.info("👆 Upload an image to begin analysis")

    # Sample analysis capabilities
    st.markdown("### 🚀 Analysis Capabilities")
    capabilities = [
        "🔍 **General Image Analysis** - Comprehensive description of image content",
        "👥 **People Detection** - Identify people, faces, emotions, and activities",
        "🏷️ **Object Recognition** - Detect and classify objects in the image",
        "📝 **Text Extraction** - Read and transcribe text within images",
        "🎨 **Style Analysis** - Analyze artistic style, colors, and composition",
        "🏢 **Scene Understanding** - Identify locations, settings, and contexts",
        "📊 **Chart Reading** - Analyze graphs, charts, and data visualizations",
        "🔒 **Content Safety** - Assess appropriateness and safety of content"
    ]

    for capability in capabilities:
        st.markdown(capability)
