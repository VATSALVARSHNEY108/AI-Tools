import streamlit as st
from utils.gemini_client import generate_text, chat_with_context, initialize_gemini_client

# Initialize Gemini client
initialize_gemini_client()

st.title("🔍 Research Tools")
st.markdown("AI-powered research assistance and knowledge extraction")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📚 Topic Research", "❓ Q&A Generator", "📄 Literature Review", "🧠 Knowledge Synthesis", "🎯 Research Planning"])

with tab1:
    st.header("Topic Research")
    st.markdown("Comprehensive research on any topic")

    research_topic = st.text_input("Enter your research topic:")

    col1, col2 = st.columns(2)
    with col1:
        research_depth = st.selectbox(
            "Research depth:",
            ["Overview", "Detailed", "Comprehensive", "Academic"]
        )
    with col2:
        research_focus = st.multiselect(
            "Focus areas:",
            ["Current Trends", "Historical Context", "Key Players", "Statistics", "Pros & Cons", "Future Outlook"]
        )

    target_audience = st.selectbox(
        "Target audience:",
        ["General Public", "Students", "Professionals", "Researchers", "Decision Makers"]
    )

    if st.button("🔍 Research Topic", type="primary"):
        if research_topic:
            with st.spinner("Conducting research..."):
                focus_text = ", ".join(research_focus) if research_focus else "general overview"
                prompt = f"""Conduct a {research_depth.lower()} research on '{research_topic}' for {target_audience.lower()}.

Focus on: {focus_text}

Please provide:
1. Executive summary
2. Key findings and insights
3. Important facts and statistics
4. Current developments
5. Implications and significance
6. Sources and further reading suggestions

Make it comprehensive yet accessible for the target audience."""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Research Results:")
                st.write(result)
        else:
            st.warning("Please enter a research topic.")

with tab2:
    st.header("Q&A Generator")
    st.markdown("Generate questions and answers for any topic")

    qa_topic = st.text_input("Enter topic for Q&A generation:")

    col1, col2 = st.columns(2)
    with col1:
        qa_type = st.selectbox(
            "Question type:",
            ["Interview Questions", "Study Questions", "FAQ", "Discussion Questions", "Quiz Questions"]
        )
    with col2:
        difficulty_level = st.selectbox(
            "Difficulty level:",
            ["Beginner", "Intermediate", "Advanced", "Expert"]
        )

    num_questions = st.slider("Number of questions:", 5, 20, 10)

    if st.button("📝 Generate Q&A", type="primary"):
        if qa_topic:
            with st.spinner("Generating questions and answers..."):
                prompt = f"""Generate {num_questions} {qa_type.lower()} about '{qa_topic}' at {difficulty_level.lower()} level.

Format each as:
Q: [Question]
A: [Detailed Answer]

Make sure questions are:
- Relevant and insightful
- Progressively challenging
- Cover different aspects of the topic
- Include both factual and analytical questions"""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Generated Q&A:")
                st.write(result)
        else:
            st.warning("Please enter a topic for Q&A generation.")

with tab3:
    st.header("Literature Review Assistant")
    st.markdown("Analyze and synthesize research literature")

    literature_topic = st.text_input("Research area for literature review:")

    col1, col2 = st.columns(2)
    with col1:
        review_type = st.selectbox(
            "Review type:",
            ["Systematic Review", "Narrative Review", "Meta-Analysis Summary", "Scoping Review"]
        )
    with col2:
        time_period = st.selectbox(
            "Time focus:",
            ["Recent (2020-2024)", "Last Decade", "Comprehensive", "Historical Perspective"]
        )

    research_questions = st.text_area(
        "Key research questions:",
        placeholder="Enter the main research questions you want to address"
    )

    if st.button("📖 Generate Literature Review", type="primary"):
        if literature_topic and research_questions:
            with st.spinner("Analyzing literature..."):
                prompt = f"""Create a {review_type.lower()} on '{literature_topic}' focusing on {time_period.lower()}.

Research Questions:
{research_questions}

Please provide:
1. Introduction and background
2. Methodology overview
3. Key themes and findings
4. Major studies and contributions
5. Gaps in current research
6. Conclusions and future directions
7. Theoretical frameworks
8. Practical implications

Structure it as an academic literature review with clear sections."""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Literature Review:")
                st.write(result)
        else:
            st.warning("Please enter both topic and research questions.")

with tab4:
    st.header("Knowledge Synthesis")
    st.markdown("Synthesize information from multiple sources or perspectives")

    synthesis_topic = st.text_input("Topic for knowledge synthesis:")

    # Multiple input sources
    st.subheader("Information Sources")
    sources = []

    for i in range(3):
        source_input = st.text_area(
            f"Source {i + 1} (optional):",
            height=100,
            key=f"source_{i}",
            placeholder="Paste information, findings, or perspectives from different sources"
        )
        if source_input.strip():
            sources.append(source_input)

    synthesis_approach = st.selectbox(
        "Synthesis approach:",
        ["Comparative Analysis", "Thematic Integration", "Consensus Building", "Critical Analysis",
         "Framework Development"]
    )

    if st.button("🧠 Synthesize Knowledge", type="primary"):
        if synthesis_topic and sources:
            with st.spinner("Synthesizing information..."):
                sources_text = "\n\n".join([f"Source {i + 1}: {source}" for i, source in enumerate(sources)])

                prompt = f"""Perform {synthesis_approach.lower()} on '{synthesis_topic}' using the following sources:

{sources_text}

Please provide:
1. Integrated summary of key insights
2. Common themes and patterns
3. Conflicting viewpoints and how to reconcile them
4. Synthesized conclusions
5. Implications and applications
6. Areas of agreement and disagreement
7. Recommendations based on synthesis

Create a coherent, unified understanding from these diverse sources."""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Knowledge Synthesis:")
                st.write(result)
        else:
            st.warning("Please enter a topic and at least one source.")

with tab5:
    st.header("Research Planning")
    st.markdown("Plan and structure your research projects")

    project_title = st.text_input("Research project title:")
    research_objective = st.text_area("Research objective/goal:")

    col1, col2 = st.columns(2)
    with col1:
        project_type = st.selectbox(
            "Project type:",
            ["Academic Research", "Market Research", "Policy Research", "Technical Research", "Creative Research"]
        )
    with col2:
        timeline = st.selectbox(
            "Timeline:",
            ["1-2 weeks", "1 month", "3 months", "6 months", "1 year+"]
        )

    resources = st.multiselect(
        "Available resources:",
        ["Academic Databases", "Surveys/Interviews", "Experiments", "Data Analysis", "Literature Review", "Field Work"]
    )

    constraints = st.text_area("Constraints or limitations:")

    if st.button("📋 Create Research Plan", type="primary"):
        if project_title and research_objective:
            with st.spinner("Creating research plan..."):
                resources_text = ", ".join(resources) if resources else "standard research methods"

                prompt = f"""Create a comprehensive research plan for:

Project: {project_title}
Type: {project_type}
Timeline: {timeline}
Objective: {research_objective}
Available Resources: {resources_text}
Constraints: {constraints}

Please provide:
1. Research methodology and approach
2. Detailed timeline with milestones
3. Literature review strategy
4. Data collection methods
5. Analysis framework
6. Expected outcomes and deliverables
7. Risk assessment and mitigation
8. Resource allocation
9. Quality assurance measures
10. Evaluation criteria

Format as a structured research proposal."""

                result = generate_text(prompt, model="gemini-2.5-pro")
                st.markdown("### Research Plan:")
                st.write(result)
        else:
            st.warning("Please enter project title and research objective.")

# Research utilities
st.markdown("---")
st.subheader("🛠️ Research Utilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Quick Research**")
    quick_query = st.text_input("Quick research query:")
    if st.button("🔍 Quick Research"):
        if quick_query:
            with st.spinner("Researching..."):
                prompt = f"Provide a concise but informative research summary on: {quick_query}"
                result = generate_text(prompt)
                st.write(result)

with col2:
    st.markdown("**Fact Check**")
    fact_claim = st.text_input("Claim to fact-check:")
    if st.button("✅ Fact Check"):
        if fact_claim:
            with st.spinner("Fact-checking..."):
                prompt = f"Fact-check this claim and provide evidence: {fact_claim}"
                result = generate_text(prompt)
                st.write(result)

# Research tips
with st.expander("💡 Research Tips & Best Practices"):
    st.markdown("""
    **Effective Research Strategies:**
    - Start with broad overview, then narrow down
    - Use multiple perspectives and sources
    - Verify information from reliable sources
    - Keep track of sources and citations
    - Look for patterns and contradictions

    **Quality Research Checklist:**
    - ✅ Clear research questions
    - ✅ Multiple reliable sources
    - ✅ Balanced perspectives
    - ✅ Current and relevant information
    - ✅ Proper methodology

    **Using These Tools:**
    - **Topic Research**: Get comprehensive overviews
    - **Q&A Generator**: Prepare for presentations or study
    - **Literature Review**: Academic research synthesis
    - **Knowledge Synthesis**: Combine multiple viewpoints
    - **Research Planning**: Structure your projects
    """)
