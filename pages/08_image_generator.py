import streamlit as st
import os
from utils.gemini_client import initialize_gemini_client
from google import genai
from google.genai import types
import io
from PIL import Image
import time

# Initialize Gemini client
initialize_gemini_client()

st.title("🎨 Advanced Image Generation")
st.markdown("Create stunning images with AI using advanced prompting techniques")


# Advanced image generation function
def generate_advanced_image(prompt, style="realistic", aspect_ratio="1:1", quality="standard", negative_prompt=""):
    """Generate image using Gemini with advanced parameters"""
    try:
        client = st.session_state.get('gemini_client')
        if not client:
            raise Exception("Gemini client not initialized")

        # Enhanced prompt engineering
        style_prompts = {
            "realistic": "photorealistic, high detail, professional photography, sharp focus",
            "artistic": "artistic, painterly, creative interpretation, expressive",
            "cinematic": "cinematic lighting, movie scene, dramatic composition, film quality",
            "anime": "anime style, manga artwork, Japanese animation, detailed character design",
            "oil_painting": "oil painting, classical art style, textured brushstrokes, museum quality",
            "digital_art": "digital art, concept art, modern illustration, vibrant colors",
            "minimalist": "minimalist design, clean lines, simple composition, modern aesthetic",
            "vintage": "vintage style, retro aesthetic, aged film look, nostalgic atmosphere"
        }

        enhanced_prompt = f"{prompt}, {style_prompts.get(style, '')}, {quality} quality"
        if negative_prompt:
            enhanced_prompt += f". Avoid: {negative_prompt}"

        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=enhanced_prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        if response.candidates:
            content = response.candidates[0].content
            if content and content.parts:
                for part in content.parts:
                    if part.inline_data and part.inline_data.data:
                        return part.inline_data.data
        return None
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return None


# Image generation interface
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Basic Generation", "🚀 Advanced Studio", "🔄 Batch Generation", "✨ Style Transfer"])

with tab1:
    st.header("Basic Image Generation")

    prompt = st.text_area("Describe your image:", height=100,
                          placeholder="A serene mountain landscape at sunset with a lake reflecting the golden sky...")

    col1, col2, col3 = st.columns(3)
    with col1:
        style = st.selectbox("Art Style:",
                             ["realistic", "artistic", "cinematic", "anime", "oil_painting", "digital_art",
                              "minimalist", "vintage"])
    with col2:
        aspect_ratio = st.selectbox("Aspect Ratio:", ["1:1", "16:9", "9:16", "4:3", "3:4"])
    with col3:
        quality = st.selectbox("Quality:", ["standard", "high", "ultra"])

    negative_prompt = st.text_input("Negative prompt (what to avoid):",
                                    placeholder="blurry, low quality, distorted...")

    if st.button("🎨 Generate Image", type="primary"):
        if prompt:
            with st.spinner("Creating your masterpiece..."):
                image_data = generate_advanced_image(prompt, style, aspect_ratio, quality, negative_prompt)
                if image_data:
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, caption=f"Generated: {prompt[:50]}...")

                    # Download button
                    st.download_button(
                        label="📥 Download Image",
                        data=image_data,
                        file_name=f"generated_image_{int(time.time())}.png",
                        mime="image/png"
                    )
        else:
            st.warning("Please enter a description for your image.")

with tab2:
    st.header("Advanced Generation Studio")

    # Advanced prompt builder
    st.subheader("🛠️ Prompt Builder")

    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Main Subject:", placeholder="a majestic eagle")
        setting = st.text_input("Setting/Environment:", placeholder="mountain peak at dawn")
        mood = st.selectbox("Mood:", ["peaceful", "dramatic", "mysterious", "energetic", "romantic", "futuristic"])

    with col2:
        camera_angle = st.selectbox("Camera Angle:",
                                    ["eye-level", "bird's eye view", "low angle", "close-up", "wide shot"])
        lighting = st.selectbox("Lighting:",
                                ["natural", "golden hour", "dramatic shadows", "soft diffused", "neon", "candlelight"])
        color_palette = st.selectbox("Color Palette:",
                                     ["natural", "vibrant", "muted", "monochrome", "warm tones", "cool tones"])

    # Auto-generate enhanced prompt
    if st.button("🔄 Auto-Generate Prompt"):
        if subject:
            auto_prompt = f"{subject} in {setting}, {mood} atmosphere, {camera_angle} perspective, {lighting} lighting, {color_palette} color scheme"
            st.session_state.auto_prompt = auto_prompt

    final_prompt = st.text_area("Final Prompt:",
                                value=st.session_state.get('auto_prompt', ''),
                                height=120)

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            creativity = st.slider("Creativity Level:", 0.1, 1.0, 0.7)
            detail_level = st.selectbox("Detail Level:", ["low", "medium", "high", "ultra-detailed"])
        with col2:
            composition = st.selectbox("Composition:",
                                       ["rule of thirds", "centered", "symmetrical", "dynamic", "abstract"])
            texture = st.selectbox("Texture:", ["smooth", "rough", "glossy", "matte", "textured"])

    if st.button("🚀 Generate Advanced Image", type="primary"):
        if final_prompt:
            with st.spinner("Creating advanced artwork..."):
                enhanced_prompt = f"{final_prompt}, {detail_level} detail, {composition} composition, {texture} texture"
                image_data = generate_advanced_image(enhanced_prompt, style="artistic")
                if image_data:
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, caption="Advanced Generation Result")

with tab3:
    st.header("Batch Generation")

    batch_prompts = st.text_area("Enter multiple prompts (one per line):", height=150,
                                 placeholder="A sunset over the ocean\nA futuristic city skyline\nA peaceful forest path")

    col1, col2 = st.columns(2)
    with col1:
        batch_style = st.selectbox("Batch Style:",
                                   ["realistic", "artistic", "cinematic", "anime", "digital_art"], key="batch_style")
    with col2:
        batch_count = st.number_input("Images per prompt:", min_value=1, max_value=5, value=1)

    if st.button("🔄 Generate Batch", type="primary"):
        if batch_prompts:
            prompts = [p.strip() for p in batch_prompts.split('\n') if p.strip()]

            if prompts:
                with st.spinner(f"Generating {len(prompts) * batch_count} images..."):
                    cols = st.columns(min(3, len(prompts)))

                    for i, prompt in enumerate(prompts):
                        col = cols[i % len(cols)]
                        with col:
                            st.write(f"**Prompt {i + 1}:** {prompt[:30]}...")

                            for j in range(batch_count):
                                image_data = generate_advanced_image(prompt, batch_style)
                                if image_data:
                                    image = Image.open(io.BytesIO(image_data))
                                    st.image(image, caption=f"Variant {j + 1}")

with tab4:
    st.header("Style Transfer & Variations")

    reference_style = st.text_input("Reference Style Description:",
                                    placeholder="Van Gogh's Starry Night style, impressionist painting...")

    base_prompt = st.text_area("Base Image Description:", height=100,
                               placeholder="A portrait of a woman in a garden...")

    col1, col2 = st.columns(2)
    with col1:
        style_strength = st.slider("Style Strength:", 0.1, 1.0, 0.7)
        variation_count = st.number_input("Number of Variations:", min_value=1, max_value=6, value=3)

    with col2:
        artistic_movement = st.selectbox("Artistic Movement:",
                                         ["impressionism", "cubism", "surrealism", "art nouveau", "pop art",
                                          "abstract expressionism"])

    if st.button("🎭 Apply Style Transfer", type="primary"):
        if base_prompt and reference_style:
            with st.spinner("Creating styled variations..."):
                styled_prompt = f"{base_prompt} in the style of {reference_style}, {artistic_movement} influence"

                cols = st.columns(min(3, variation_count))
                for i in range(variation_count):
                    col = cols[i % len(cols)]
                    with col:
                        variation_prompt = f"{styled_prompt}, variation {i + 1}"
                        image_data = generate_advanced_image(variation_prompt, "artistic")
                        if image_data:
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption=f"Style Variation {i + 1}")

# Image history and gallery
st.markdown("---")
st.subheader("🖼️ Generation History")

if 'image_history' not in st.session_state:
    st.session_state.image_history = []

if st.session_state.image_history:
    cols = st.columns(4)
    for i, img_data in enumerate(st.session_state.image_history[-8:]):  # Show last 8 images
        col = cols[i % 4]
        with col:
            st.image(img_data['image'], caption=img_data['prompt'][:30])
else:
    st.info("Your generated images will appear here for easy access.")

# Tips and tutorials
with st.expander("💡 Advanced Prompting Tips"):
    st.markdown("""
    **Professional Prompting Techniques:**

    **🎯 Composition Keywords:**
    - "rule of thirds", "golden ratio", "leading lines"
    - "symmetrical", "asymmetrical", "balanced composition"

    **📸 Photography Terms:**
    - "shallow depth of field", "bokeh", "macro photography"
    - "long exposure", "HDR", "golden hour lighting"

    **🎨 Artistic Styles:**
    - "hyperrealistic", "photorealistic", "oil painting"
    - "watercolor", "digital art", "concept art"

    **🌟 Quality Enhancers:**
    - "8K resolution", "ultra-detailed", "sharp focus"
    - "professional photography", "award-winning"

    **⚡ Advanced Techniques:**
    - Use negative prompts to avoid unwanted elements
    - Combine multiple styles for unique results
    - Specify camera settings for photorealistic images
    - Use artistic movements for creative interpretations
    """)