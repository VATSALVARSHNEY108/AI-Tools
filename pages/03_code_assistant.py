import streamlit as st
from utils.gemini_client import generate_text, initialize_gemini_client

# Initialize Gemini client
initialize_gemini_client()

st.title("💻 Code Assistant")
st.markdown("AI-powered coding help, generation, and debugging")

tab1, tab2, tab3, tab4 = st.tabs(["⚡ Code Generation", "🐛 Debug & Fix", "📖 Code Explanation", "🔄 Code Conversion"])

with tab1:
    st.header("Code Generation")

    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Programming Language:",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift"]
        )
    with col2:
        code_type = st.selectbox(
            "Code Type:",
            ["Function", "Class", "Script", "API", "Algorithm", "Data Structure", "Web Component"]
        )

    description = st.text_area("Describe what you want the code to do:")
    requirements = st.text_area("Specific requirements or constraints:")

    col1, col2 = st.columns(2)
    with col1:
        include_comments = st.checkbox("Include comments", value=True)
    with col2:
        include_tests = st.checkbox("Include example usage/tests")

    if st.button("Generate Code", type="primary"):
        if description:
            with st.spinner("Generating code..."):
                prompt = f"""Generate a {code_type.lower()} in {language} that {description}.

Requirements: {requirements}
{"Include detailed comments explaining the code." if include_comments else ""}
{"Include example usage or test cases." if include_tests else ""}

Please provide clean, well-structured, and production-ready code."""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Generated Code:")
                st.code(result, language=language.lower())
        else:
            st.warning("Please describe what you want the code to do.")

with tab2:
    st.header("Debug & Fix Code")

    code_to_debug = st.text_area("Paste your code here:", height=200)

    col1, col2 = st.columns(2)
    with col1:
        debug_language = st.selectbox(
            "Language:",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift"],
            key="debug_lang"
        )
    with col2:
        error_message = st.text_input("Error message (if any):")

    issue_description = st.text_area("Describe the issue or expected behavior:")

    if st.button("Debug Code", type="primary"):
        if code_to_debug:
            with st.spinner("Analyzing and debugging code..."):
                prompt = f"""Analyze this {debug_language} code and help debug it:

```{debug_language.lower()}
{code_to_debug}
```

Error message: {error_message}
Issue description: {issue_description}

Please provide:
1. Analysis of the problem
2. Corrected code
3. Explanation of the fix
4. Best practices to avoid similar issues"""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Debug Results:")
                st.write(result)
        else:
            st.warning("Please paste the code you want to debug.")

with tab3:
    st.header("Code Explanation")

    code_to_explain = st.text_area("Paste code to explain:", height=200)

    col1, col2 = st.columns(2)
    with col1:
        explain_language = st.selectbox(
            "Language:",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift"],
            key="explain_lang"
        )
    with col2:
        explanation_level = st.selectbox(
            "Explanation level:",
            ["Beginner", "Intermediate", "Advanced"]
        )

    if st.button("Explain Code", type="primary"):
        if code_to_explain:
            with st.spinner("Analyzing code..."):
                prompt = f"""Explain this {explain_language} code for a {explanation_level.lower()} level programmer:

```{explain_language.lower()}
{code_to_explain}
```

Please provide:
1. Overview of what the code does
2. Line-by-line explanation
3. Key concepts used
4. Potential improvements
5. Common use cases"""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Code Explanation:")
                st.write(result)
        else:
            st.warning("Please paste code to explain.")

with tab4:
    st.header("Code Conversion")

    source_code = st.text_area("Paste source code:", height=200)

    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox(
            "From language:",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift"],
            key="source_lang"
        )
    with col2:
        target_lang = st.selectbox(
            "To language:",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift"],
            key="target_lang"
        )

    if st.button("Convert Code", type="primary"):
        if source_code and source_lang != target_lang:
            with st.spinner("Converting code..."):
                prompt = f"""Convert this {source_lang} code to {target_lang}:

```{source_lang.lower()}
{source_code}
```

Please provide:
1. Converted code with proper syntax
2. Explanation of key differences
3. Notes about language-specific features
4. Best practices for the target language"""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Converted Code:")
                st.write(result)
        else:
            if not source_code:
                st.warning("Please paste source code to convert.")
            if source_lang == target_lang:
                st.warning("Please select different source and target languages.")

# Additional options
st.markdown("---")
with st.expander("⚙️ Advanced Options"):
    model_choice = st.selectbox("AI Model:", ["gemini-2.5-flash", "gemini-2.5-pro"], key="code_model")
    temperature = st.slider("Creativity Level:", 0.1, 1.0, 0.7, 0.1, key="code_temp")

    st.info("Higher creativity levels produce more varied and creative coding solutions.")
