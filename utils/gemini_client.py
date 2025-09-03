import os
import streamlit as st
from google import genai
from google.genai import types
import logging


def initialize_gemini_client():
    """Initialize the Gemini client and store in session state"""
    if 'gemini_client' not in st.session_state:
        try:
            api_key = os.getenv("GEMINI_API_KEY", "default_key")
            st.session_state.gemini_client = genai.Client(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {e}")
            return False
    return True


def get_gemini_client():
    """Get the Gemini client from session state"""
    return st.session_state.get('gemini_client')


def generate_text(prompt, model="gemini-2.5-flash", temperature=0.7):
    """Generate text using Gemini"""
    try:
        client = get_gemini_client()
        if not client:
            raise Exception("Gemini client not initialized")

        config = types.GenerateContentConfig(
            temperature=temperature
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        return response.text if response.text else "No response generated"
    except Exception as e:
        logging.error(f"Text generation error: {e}")
        return f"Error generating text: {e}"


def analyze_image_with_prompt(image_bytes, prompt, mime_type="image/jpeg"):
    """Analyze image with custom prompt using Gemini Vision"""
    try:
        client = get_gemini_client()
        if not client:
            raise Exception("Gemini client not initialized")

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt
            ],
        )

        return response.text if response.text else "No analysis generated"
    except Exception as e:
        logging.error(f"Image analysis error: {e}")
        return f"Error analyzing image: {e}"


def chat_with_context(message, context="", model="gemini-2.5-flash"):
    """Chat with Gemini with optional context"""
    try:
        client = get_gemini_client()
        if not client:
            raise Exception("Gemini client not initialized")

        if context:
            full_prompt = f"Context: {context}\n\nUser: {message}"
        else:
            full_prompt = message

        response = client.models.generate_content(
            model=model,
            contents=full_prompt
        )

        return response.text if response.text else "No response generated"
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return f"Error in chat: {e}"
