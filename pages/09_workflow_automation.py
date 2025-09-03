import streamlit as st
import json
import time
from utils.gemini_client import generate_text, chat_with_context, analyze_image_with_prompt, initialize_gemini_client
from utils.file_processors import process_text_file, process_pdf_file, process_docx_file, process_csv_file
import pandas as pd
import plotly.express as px

# Initialize Gemini client
initialize_gemini_client()

st.title("⚡ AI Workflow Automation")
st.markdown("Create and execute complex AI workflows with multiple steps and conditional logic")

# Initialize session state
if 'workflows' not in st.session_state:
    st.session_state.workflows = {}
if 'workflow_results' not in st.session_state:
    st.session_state.workflow_results = {}

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔧 Workflow Builder", "▶️ Execute Workflows", "📊 Results Dashboard", "🤖 Smart Templates"])

with tab1:
    st.header("Workflow Builder")

    # Workflow creation
    workflow_name = st.text_input("Workflow Name:", placeholder="My AI Workflow")
    workflow_description = st.text_area("Description:", placeholder="Describe what this workflow does...")

    if workflow_name:
        st.subheader(f"Building: {workflow_name}")

        # Step builder
        st.markdown("### Workflow Steps")

        if f"{workflow_name}_steps" not in st.session_state:
            st.session_state[f"{workflow_name}_steps"] = []

        # Add step interface
        with st.expander("➕ Add New Step"):
            step_type = st.selectbox("Step Type:", [
                "Text Generation", "Image Analysis", "Document Processing",
                "Data Analysis", "Code Generation", "Conditional Logic",
                "Data Transformation", "Multi-Modal Analysis"
            ])

            step_name = st.text_input("Step Name:",
                                      placeholder=f"Step {len(st.session_state[f'{workflow_name}_steps']) + 1}")

            # Step-specific configuration
            if step_type == "Text Generation":
                prompt_template = st.text_area("Prompt Template:",
                                               placeholder="Generate a summary of: {input_text}")
                model = st.selectbox("Model:", ["gemini-2.5-flash", "gemini-2.5-pro"])

            elif step_type == "Image Analysis":
                analysis_prompt = st.text_area("Analysis Prompt:",
                                               placeholder="Analyze this image and describe: {focus_area}")

            elif step_type == "Document Processing":
                processing_type = st.selectbox("Processing:", ["Summarize", "Extract", "Analyze", "Q&A"])

            elif step_type == "Data Analysis":
                analysis_type = st.selectbox("Analysis:", ["Statistics", "Visualization", "Insights", "Correlation"])

            elif step_type == "Conditional Logic":
                condition = st.text_input("Condition:", placeholder="if {previous_result} contains 'positive'")
                true_action = st.text_input("If True:", placeholder="next_step_id")
                false_action = st.text_input("If False:", placeholder="alternative_step_id")

            if st.button("Add Step"):
                step_config = {
                    "id": len(st.session_state[f"{workflow_name}_steps"]),
                    "name": step_name,
                    "type": step_type,
                    "config": {}
                }

                if step_type == "Text Generation":
                    step_config["config"] = {"prompt": prompt_template, "model": model}
                elif step_type == "Image Analysis":
                    step_config["config"] = {"prompt": analysis_prompt}
                elif step_type == "Document Processing":
                    step_config["config"] = {"type": processing_type}
                elif step_type == "Data Analysis":
                    step_config["config"] = {"type": analysis_type}
                elif step_type == "Conditional Logic":
                    step_config["config"] = {"condition": condition, "true": true_action, "false": false_action}

                st.session_state[f"{workflow_name}_steps"].append(step_config)
                st.rerun()

        # Display current steps
        if st.session_state[f"{workflow_name}_steps"]:
            st.markdown("### Current Steps")
            for i, step in enumerate(st.session_state[f"{workflow_name}_steps"]):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{i + 1}. {step['name']}** ({step['type']})")
                    with col2:
                        if st.button("Edit", key=f"edit_{i}"):
                            st.session_state[f"editing_{i}"] = True
                    with col3:
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state[f"{workflow_name}_steps"].pop(i)
                            st.rerun()

        # Save workflow
        if st.button("💾 Save Workflow", type="primary"):
            if st.session_state[f"{workflow_name}_steps"]:
                st.session_state.workflows[workflow_name] = {
                    "description": workflow_description,
                    "steps": st.session_state[f"{workflow_name}_steps"].copy(),
                    "created": time.time()
                }
                st.success(f"Workflow '{workflow_name}' saved!")
            else:
                st.warning("Add at least one step before saving.")

with tab2:
    st.header("Execute Workflows")

    if st.session_state.workflows:
        selected_workflow = st.selectbox("Select Workflow:", list(st.session_state.workflows.keys()))

        if selected_workflow:
            workflow = st.session_state.workflows[selected_workflow]
            st.write(f"**Description:** {workflow['description']}")

            # Input configuration
            st.subheader("Input Configuration")

            # Multiple input types
            input_type = st.selectbox("Input Type:", ["Text", "File Upload", "Multiple Files", "Manual Input"])

            inputs = {}

            if input_type == "Text":
                inputs["text"] = st.text_area("Input Text:", height=150)

            elif input_type == "File Upload":
                uploaded_file = st.file_uploader("Upload File:", type=['txt', 'pdf', 'docx', 'csv', 'jpg', 'png'])
                if uploaded_file:
                    inputs["file"] = uploaded_file

            elif input_type == "Multiple Files":
                uploaded_files = st.file_uploader("Upload Files:", accept_multiple_files=True)
                if uploaded_files:
                    inputs["files"] = uploaded_files

            # Execution parameters
            with st.expander("⚙️ Execution Settings"):
                parallel_execution = st.checkbox("Parallel Execution (where possible)")
                save_intermediate = st.checkbox("Save Intermediate Results", value=True)
                auto_retry = st.checkbox("Auto-retry on Failure")

            # Execute workflow
            if st.button("▶️ Execute Workflow", type="primary"):
                if inputs:
                    execute_workflow(selected_workflow, workflow, inputs, parallel_execution, save_intermediate)
                else:
                    st.warning("Please provide input for the workflow.")
    else:
        st.info("No workflows created yet. Use the Workflow Builder to create your first workflow.")


def execute_workflow(name, workflow, inputs, parallel=False, save_intermediate=True):
    """Execute a workflow with given inputs"""
    with st.spinner(f"Executing workflow: {name}"):
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = workflow["steps"]
        total_steps = len(steps)

        for i, step in enumerate(steps):
            status_text.text(f"Executing step {i + 1}/{total_steps}: {step['name']}")

            try:
                # Execute step based on type
                if step["type"] == "Text Generation":
                    prompt = step["config"]["prompt"]
                    model = step["config"].get("model", "gemini-2.5-flash")

                    # Replace placeholders with previous results or inputs
                    if "{input_text}" in prompt and "text" in inputs:
                        prompt = prompt.replace("{input_text}", inputs["text"])

                    result = generate_text(prompt, model)

                elif step["type"] == "Image Analysis":
                    if "file" in inputs and inputs["file"].type.startswith("image"):
                        image_bytes = inputs["file"].read()
                        analysis_prompt = step["config"]["prompt"]
                        result = analyze_image_with_prompt(image_bytes, analysis_prompt)
                    else:
                        result = "No image file provided"

                elif step["type"] == "Document Processing":
                    if "file" in inputs:
                        file_type = inputs["file"].name.split(".")[-1].lower()
                        if file_type == "pdf":
                            content = process_pdf_file(inputs["file"])
                        elif file_type == "docx":
                            content = process_docx_file(inputs["file"])
                        else:
                            content = process_text_file(inputs["file"])

                        processing_type = step["config"]["type"]
                        if processing_type == "Summarize":
                            result = generate_text(f"Summarize this document: {content}")
                        elif processing_type == "Extract":
                            result = generate_text(f"Extract key information from: {content}")
                        else:
                            result = content
                    else:
                        result = "No file provided"

                results[step["name"]] = result

                if save_intermediate:
                    st.write(f"**{step['name']} Result:**")
                    st.write(result)
                    st.markdown("---")

            except Exception as e:
                st.error(f"Error in step '{step['name']}': {e}")
                results[step["name"]] = f"Error: {e}"

            progress_bar.progress((i + 1) / total_steps)

        # Save results
        st.session_state.workflow_results[f"{name}_{int(time.time())}"] = {
            "workflow": name,
            "results": results,
            "timestamp": time.time(),
            "inputs": str(inputs)
        }

        status_text.text("Workflow execution completed!")
        st.success("✅ Workflow executed successfully!")


with tab3:
    st.header("Results Dashboard")

    if st.session_state.workflow_results:
        # Results overview
        st.subheader("Execution History")

        results_df = pd.DataFrame([
            {
                "Execution ID": key,
                "Workflow": result["workflow"],
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result["timestamp"])),
                "Steps Completed": len(result["results"])
            }
            for key, result in st.session_state.workflow_results.items()
        ])

        st.dataframe(results_df, use_container_width=True)

        # Detailed results
        selected_execution = st.selectbox("View Detailed Results:", list(st.session_state.workflow_results.keys()))

        if selected_execution:
            execution_result = st.session_state.workflow_results[selected_execution]

            st.subheader(f"Results for: {execution_result['workflow']}")

            for step_name, result in execution_result["results"].items():
                with st.expander(f"📋 {step_name}"):
                    st.write(result)

            # Export results
            if st.button("📥 Export Results"):
                export_data = json.dumps(execution_result, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"workflow_results_{selected_execution}.json",
                    mime="application/json"
                )
    else:
        st.info("No workflow executions yet.")

with tab4:
    st.header("Smart Workflow Templates")

    templates = {
        "Content Creation Pipeline": {
            "description": "Generate, refine, and optimize content across multiple formats",
            "steps": [
                {"name": "Research Topic", "type": "Text Generation",
                 "config": {"prompt": "Research and provide comprehensive information about: {topic}"}},
                {"name": "Create Outline", "type": "Text Generation",
                 "config": {"prompt": "Create a detailed outline for: {research_result}"}},
                {"name": "Write Content", "type": "Text Generation",
                 "config": {"prompt": "Write engaging content based on: {outline}"}},
                {"name": "Optimize for SEO", "type": "Text Generation",
                 "config": {"prompt": "Optimize this content for SEO: {content}"}}
            ]
        },
        "Document Intelligence": {
            "description": "Extract, analyze, and synthesize information from documents",
            "steps": [
                {"name": "Extract Text", "type": "Document Processing", "config": {"type": "Extract"}},
                {"name": "Summarize", "type": "Text Generation",
                 "config": {"prompt": "Create executive summary: {extracted_text}"}},
                {"name": "Extract Entities", "type": "Text Generation",
                 "config": {"prompt": "Extract all names, dates, and key entities: {extracted_text}"}},
                {"name": "Generate Insights", "type": "Text Generation",
                 "config": {"prompt": "Provide strategic insights from: {summary}"}}
            ]
        },
        "Data Analysis Pipeline": {
            "description": "Complete data analysis from upload to insights",
            "steps": [
                {"name": "Load Data", "type": "Data Analysis", "config": {"type": "Statistics"}},
                {"name": "Clean Data", "type": "Data Transformation", "config": {"type": "cleaning"}},
                {"name": "Generate Visualizations", "type": "Data Analysis", "config": {"type": "Visualization"}},
                {"name": "AI Insights", "type": "Text Generation",
                 "config": {"prompt": "Analyze this data and provide business insights: {data_summary}"}}
            ]
        },
        "Multi-Modal Analysis": {
            "description": "Analyze images, text, and data together for comprehensive insights",
            "steps": [
                {"name": "Analyze Image", "type": "Image Analysis",
                 "config": {"prompt": "Describe all elements in this image"}},
                {"name": "Process Text", "type": "Text Generation",
                 "config": {"prompt": "Analyze this text: {input_text}"}},
                {"name": "Correlate Findings", "type": "Multi-Modal Analysis", "config": {"type": "correlation"}},
                {"name": "Generate Report", "type": "Text Generation",
                 "config": {"prompt": "Create comprehensive report from: {image_analysis} and {text_analysis}"}}
            ]
        }
    }

    for template_name, template in templates.items():
        with st.expander(f"📋 {template_name}"):
            st.write(template["description"])
            st.write(f"**Steps:** {len(template['steps'])}")

            if st.button(f"Use {template_name}", key=f"use_{template_name}"):
                st.session_state.workflows[template_name] = {
                    "description": template["description"],
                    "steps": template["steps"],
                    "created": time.time()
                }
                st.success(f"Template '{template_name}' added to your workflows!")

# Advanced features
st.markdown("---")
st.subheader("🚀 Advanced Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔄 Workflow Chaining**")
    st.write("Connect multiple workflows to create complex automation pipelines")

with col2:
    st.markdown("**⏱️ Scheduling**")
    st.write("Schedule workflows to run automatically at specified times")

with col3:
    st.markdown("**📊 Analytics**")
    st.write("Track workflow performance and optimization opportunities")

# Tips and best practices
with st.expander("💡 Workflow Optimization Tips"):
    st.markdown("""
    **Best Practices for AI Workflows:**

    **🎯 Design Principles:**
    - Keep steps focused and single-purpose
    - Use clear, descriptive names for steps
    - Plan for error handling and fallbacks

    **⚡ Performance Tips:**
    - Use parallel execution for independent steps
    - Cache intermediate results when possible
    - Optimize prompts for clarity and efficiency

    **🔧 Advanced Techniques:**
    - Use conditional logic for dynamic workflows
    - Implement feedback loops for iterative improvement
    - Create reusable step templates
    - Monitor execution times and optimize bottlenecks

    **📈 Scaling Workflows:**
    - Test with small datasets first
    - Implement batch processing for large inputs
    - Use appropriate AI models for each task
    - Monitor API usage and costs
    """)