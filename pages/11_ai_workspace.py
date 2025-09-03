import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from utils.gemini_client import generate_text, initialize_gemini_client
import plotly.express as px
import plotly.graph_objects as go

# Initialize Gemini client
initialize_gemini_client()

st.title("🚀 AI Collaborative Workspace")
st.markdown("Manage projects, collaborate with AI, and track progress in one intelligent workspace")

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'tasks' not in st.session_state:
    st.session_state.tasks = {}
if 'ai_conversations' not in st.session_state:
    st.session_state.ai_conversations = {}
if 'workspace_analytics' not in st.session_state:
    st.session_state.workspace_analytics = {"sessions": 0, "ai_interactions": 0}

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Project Dashboard", "🤖 AI Assistant", "📊 Analytics", "🔄 Automation", "⚙️ Workspace Settings"])

with tab1:
    st.header("Project Management Dashboard")

    # Project creation and selection
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Projects")

        # Create new project
        with st.expander("➕ Create New Project"):
            project_name = st.text_input("Project Name:")
            project_description = st.text_area("Description:")
            project_type = st.selectbox("Project Type:",
                                        ["Research", "Development", "Content Creation", "Data Analysis", "Marketing"])

            col1_inner, col2_inner = st.columns(2)
            with col1_inner:
                priority = st.selectbox("Priority:", ["Low", "Medium", "High", "Critical"])
                due_date = st.date_input("Due Date:")

            with col2_inner:
                team_size = st.number_input("Team Size:", min_value=1, max_value=20, value=1)
                budget = st.number_input("Budget ($):", min_value=0, value=1000)

            if st.button("Create Project", type="primary"):
                if project_name:
                    project_id = f"proj_{int(time.time())}"
                    st.session_state.projects[project_id] = {
                        "name": project_name,
                        "description": project_description,
                        "type": project_type,
                        "priority": priority,
                        "due_date": str(due_date),
                        "team_size": team_size,
                        "budget": budget,
                        "created": time.time(),
                        "status": "Planning",
                        "progress": 0,
                        "tasks": []
                    }
                    st.success(f"Project '{project_name}' created!")
                    st.rerun()

    with col2:
        st.subheader("Quick Stats")
        total_projects = len(st.session_state.projects)
        active_projects = sum(
            1 for p in st.session_state.projects.values() if p['status'] in ['Planning', 'In Progress'])
        completed_projects = sum(1 for p in st.session_state.projects.values() if p['status'] == 'Completed')

        st.metric("Total Projects", total_projects)
        st.metric("Active Projects", active_projects)
        st.metric("Completed", completed_projects)

    # Project list and management
    if st.session_state.projects:
        st.subheader("Current Projects")

        # Convert projects to DataFrame for display
        projects_data = []
        for proj_id, proj in st.session_state.projects.items():
            projects_data.append({
                "ID": proj_id,
                "Name": proj["name"],
                "Type": proj["type"],
                "Priority": proj["priority"],
                "Status": proj["status"],
                "Progress": f"{proj['progress']}%",
                "Due Date": proj["due_date"]
            })

        projects_df = pd.DataFrame(projects_data)

        # Display projects with selection
        selected_project = st.selectbox("Select Project:",
                                        options=list(st.session_state.projects.keys()),
                                        format_func=lambda x: st.session_state.projects[x]["name"])

        if selected_project:
            project = st.session_state.projects[selected_project]

            # Project details
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Priority", project["priority"])
                new_status = st.selectbox("Update Status:",
                                          ["Planning", "In Progress", "Review", "Completed", "On Hold"],
                                          index=["Planning", "In Progress", "Review", "Completed", "On Hold"].index(
                                              project["status"]))
                if new_status != project["status"]:
                    st.session_state.projects[selected_project]["status"] = new_status
                    st.success("Status updated!")

            with col2:
                st.metric("Progress", f"{project['progress']}%")
                new_progress = st.slider("Update Progress:", 0, 100, project["progress"])
                if new_progress != project["progress"]:
                    st.session_state.projects[selected_project]["progress"] = new_progress

            with col3:
                days_until_due = (datetime.strptime(project["due_date"], "%Y-%m-%d") - datetime.now()).days
                st.metric("Days Until Due", days_until_due)
                if days_until_due < 7:
                    st.warning("⚠️ Due soon!")

            # Task management for project
            st.subheader(f"Tasks for {project['name']}")

            # Add new task
            with st.expander("➕ Add Task"):
                task_name = st.text_input("Task Name:")
                task_description = st.text_area("Task Description:")
                task_priority = st.selectbox("Task Priority:", ["Low", "Medium", "High"])

                if st.button("Add Task"):
                    if task_name:
                        task_id = f"task_{int(time.time())}"
                        new_task = {
                            "name": task_name,
                            "description": task_description,
                            "priority": task_priority,
                            "status": "Todo",
                            "created": time.time(),
                            "project_id": selected_project
                        }
                        st.session_state.tasks[task_id] = new_task
                        st.session_state.projects[selected_project]["tasks"].append(task_id)
                        st.success("Task added!")
                        st.rerun()

            # Display project tasks
            project_tasks = [st.session_state.tasks[task_id] for task_id in project["tasks"] if
                             task_id in st.session_state.tasks]

            if project_tasks:
                for i, task in enumerate(project_tasks):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            st.write(f"**{task['name']}** - {task['description']}")
                            st.write(f"Priority: {task['priority']} | Status: {task['status']}")

                        with col2:
                            new_task_status = st.selectbox("Status:",
                                                           ["Todo", "In Progress", "Done"],
                                                           index=["Todo", "In Progress", "Done"].index(task["status"]),
                                                           key=f"task_status_{i}")
                            task['status'] = new_task_status

                        with col3:
                            if st.button("🤖 AI Help", key=f"ai_help_{i}"):
                                ai_suggestion = generate_ai_task_suggestions(task, project)
                                st.write(ai_suggestion)

                        st.markdown("---")


def generate_ai_task_suggestions(task, project):
    """Generate AI suggestions for task completion"""
    prompt = f"""
    As a project management AI assistant, provide helpful suggestions for this task:

    Task: {task['name']}
    Description: {task['description']}
    Priority: {task['priority']}
    Status: {task['status']}

    Project Context:
    Project: {project['name']}
    Type: {project['type']}
    Description: {project['description']}

    Please provide:
    1. Specific action steps to complete this task
    2. Potential challenges and how to overcome them
    3. Resources or tools that might be helpful
    4. Estimated time to complete
    5. Dependencies on other tasks (if any)
    """

    return generate_text(prompt, model="gemini-2.5-pro")


with tab2:
    st.header("AI Project Assistant")

    # AI assistant interface
    st.subheader("💬 Chat with your AI Project Assistant")

    # Context setting
    if st.session_state.projects:
        selected_context_project = st.selectbox("Project Context:",
                                                options=["General"] + list(st.session_state.projects.keys()),
                                                format_func=lambda x: "General Assistant" if x == "General" else
                                                st.session_state.projects[x]["name"])
    else:
        selected_context_project = "General"

    # AI conversation history
    if 'ai_workspace_chat' not in st.session_state:
        st.session_state.ai_workspace_chat = []

    # Display chat history
    for message in st.session_state.ai_workspace_chat:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask your AI assistant anything about your projects...")

    if user_input:
        # Add user message
        st.session_state.ai_workspace_chat.append({"role": "user", "content": user_input})

        # Generate AI response with context
        context = ""
        if selected_context_project != "General":
            project = st.session_state.projects[selected_context_project]
            context = f"""
            Current Project Context:
            Project: {project['name']}
            Type: {project['type']}
            Description: {project['description']}
            Status: {project['status']}
            Progress: {project['progress']}%
            Priority: {project['priority']}
            """

        # Add workspace context
        workspace_context = f"""
        Workspace Context:
        Total Projects: {len(st.session_state.projects)}
        Total Tasks: {len(st.session_state.tasks)}

        You are an AI project management assistant. Help with:
        - Project planning and strategy
        - Task prioritization and breakdown
        - Resource allocation
        - Timeline management
        - Problem-solving
        """

        full_context = workspace_context + context
        ai_response = generate_text(f"Context: {full_context}\n\nUser question: {user_input}", model="gemini-2.5-pro")

        # Add AI response
        st.session_state.ai_workspace_chat.append({"role": "assistant", "content": ai_response})

        # Update analytics
        st.session_state.workspace_analytics["ai_interactions"] += 1

        st.rerun()

    # Quick AI actions
    st.subheader("🚀 Quick AI Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Generate Project Plan"):
            if st.session_state.projects:
                latest_project_id = max(st.session_state.projects.keys(),
                                        key=lambda x: st.session_state.projects[x]["created"])
                project = st.session_state.projects[latest_project_id]

                plan_prompt = f"""
                Create a detailed project plan for:
                Project: {project['name']}
                Type: {project['type']}
                Description: {project['description']}
                Team Size: {project['team_size']}
                Budget: ${project['budget']}

                Include:
                1. Project phases and milestones
                2. Task breakdown with priorities
                3. Resource allocation
                4. Risk assessment
                5. Timeline recommendations
                """

                plan = generate_text(plan_prompt, model="gemini-2.5-pro")
                st.write("### 📋 AI-Generated Project Plan")
                st.write(plan)

    with col2:
        if st.button("⚡ Optimize Workflow"):
            if st.session_state.projects and st.session_state.tasks:
                optimization_prompt = f"""
                Analyze current workspace and suggest optimizations:

                Projects: {len(st.session_state.projects)}
                Tasks: {len(st.session_state.tasks)}

                Project Status Summary:
                {json.dumps({pid: {"name": p["name"], "status": p["status"], "progress": p["progress"]}
                             for pid, p in st.session_state.projects.items()}, indent=2)}

                Provide:
                1. Workflow optimization suggestions
                2. Task prioritization recommendations
                3. Resource reallocation ideas
                4. Bottleneck identification
                5. Productivity improvements
                """

                optimization = generate_text(optimization_prompt, model="gemini-2.5-pro")
                st.write("### ⚡ Workflow Optimization")
                st.write(optimization)

    with col3:
        if st.button("📊 Generate Report"):
            if st.session_state.projects:
                report_prompt = f"""
                Generate a comprehensive workspace report:

                Workspace Overview:
                - Total Projects: {len(st.session_state.projects)}
                - Total Tasks: {len(st.session_state.tasks)}
                - AI Interactions: {st.session_state.workspace_analytics["ai_interactions"]}

                Project Details:
                {json.dumps({pid: p for pid, p in st.session_state.projects.items()}, indent=2, default=str)}

                Include:
                1. Executive summary
                2. Project status overview
                3. Key achievements
                4. Challenges and risks
                5. Next steps and recommendations
                """

                report = generate_text(report_prompt, model="gemini-2.5-pro")
                st.write("### 📊 Workspace Report")
                st.write(report)

with tab3:
    st.header("Workspace Analytics")

    if st.session_state.projects:
        # Project status distribution
        status_counts = {}
        for project in st.session_state.projects.values():
            status = project["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        col1, col2 = st.columns(2)

        with col1:
            # Status pie chart
            fig_status = px.pie(values=list(status_counts.values()),
                                names=list(status_counts.keys()),
                                title="Project Status Distribution")
            st.plotly_chart(fig_status, use_container_width=True)

        with col2:
            # Priority distribution
            priority_counts = {}
            for project in st.session_state.projects.values():
                priority = project["priority"]
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

            fig_priority = px.bar(x=list(priority_counts.keys()),
                                  y=list(priority_counts.values()),
                                  title="Project Priority Distribution")
            st.plotly_chart(fig_priority, use_container_width=True)

        # Progress overview
        st.subheader("📈 Project Progress Overview")

        progress_data = []
        for pid, project in st.session_state.projects.items():
            progress_data.append({
                "Project": project["name"],
                "Progress": project["progress"],
                "Status": project["status"],
                "Priority": project["priority"]
            })

        progress_df = pd.DataFrame(progress_data)

        fig_progress = px.bar(progress_df, x="Project", y="Progress",
                              color="Priority", title="Project Progress by Priority")
        st.plotly_chart(fig_progress, use_container_width=True)

        # Performance metrics
        st.subheader("🎯 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_progress = sum(p["progress"] for p in st.session_state.projects.values()) / len(
                st.session_state.projects)
            st.metric("Average Progress", f"{avg_progress:.1f}%")

        with col2:
            high_priority = sum(1 for p in st.session_state.projects.values() if p["priority"] == "High")
            st.metric("High Priority Projects", high_priority)

        with col3:
            overdue_projects = 0
            for project in st.session_state.projects.values():
                due_date = datetime.strptime(project["due_date"], "%Y-%m-%d")
                if due_date < datetime.now() and project["status"] != "Completed":
                    overdue_projects += 1
            st.metric("Overdue Projects", overdue_projects)

        with col4:
            st.metric("AI Interactions", st.session_state.workspace_analytics["ai_interactions"])

    else:
        st.info("Create some projects to see analytics")

with tab4:
    st.header("AI Automation Center")

    st.subheader("🔄 Automated Workflows")

    # Automation rules
    automation_rules = [
        {
            "name": "Progress Tracker",
            "description": "Automatically update project status based on task completion",
            "trigger": "Task completion",
            "action": "Update project progress"
        },
        {
            "name": "Deadline Alert",
            "description": "Send AI-generated reminders for approaching deadlines",
            "trigger": "3 days before due date",
            "action": "Generate reminder with suggestions"
        },
        {
            "name": "Resource Optimizer",
            "description": "Suggest resource reallocation based on project priorities",
            "trigger": "Weekly analysis",
            "action": "Generate optimization report"
        }
    ]

    for rule in automation_rules:
        with st.expander(f"⚙️ {rule['name']}"):
            st.write(f"**Description:** {rule['description']}")
            st.write(f"**Trigger:** {rule['trigger']}")
            st.write(f"**Action:** {rule['action']}")

            col1, col2 = st.columns(2)
            with col1:
                enabled = st.checkbox("Enable", key=f"auto_{rule['name']}")
            with col2:
                if st.button("Test Run", key=f"test_{rule['name']}"):
                    st.info(f"Testing {rule['name']} automation...")

    # Custom automation
    st.subheader("🛠️ Custom Automation")

    with st.expander("➕ Create Custom Automation"):
        auto_name = st.text_input("Automation Name:")
        auto_trigger = st.selectbox("Trigger:",
                                    ["Task Created", "Task Completed", "Project Status Change",
                                     "Due Date Approaching", "Progress Milestone", "Custom Schedule"])
        auto_action = st.text_area("AI Action Description:",
                                   placeholder="Describe what the AI should do when triggered...")

        if st.button("Create Automation"):
            if auto_name and auto_action:
                st.success(f"Automation '{auto_name}' created!")

with tab5:
    st.header("Workspace Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ General Settings")

        # Workspace preferences
        default_project_type = st.selectbox("Default Project Type:",
                                            ["Research", "Development", "Content Creation", "Data Analysis",
                                             "Marketing"])

        ai_assistant_style = st.selectbox("AI Assistant Style:",
                                          ["Professional", "Casual", "Technical", "Creative"])

        notification_frequency = st.selectbox("Notification Frequency:",
                                              ["Real-time", "Hourly", "Daily", "Weekly"])

        auto_save = st.checkbox("Auto-save progress", value=True)

        st.subheader("🎨 Interface")

        theme_preference = st.selectbox("Theme:", ["Light", "Dark", "Auto"])
        compact_view = st.checkbox("Compact view")

    with col2:
        st.subheader("📊 Data Management")

        # Export options
        if st.button("📥 Export All Data"):
            export_data = {
                "projects": st.session_state.projects,
                "tasks": st.session_state.tasks,
                "analytics": st.session_state.workspace_analytics,
                "export_date": datetime.now().isoformat()
            }

            st.download_button(
                label="Download Workspace Data",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"workspace_backup_{int(time.time())}.json",
                mime="application/json"
            )

        # Import data
        uploaded_backup = st.file_uploader("Import Workspace Data:", type="json")
        if uploaded_backup:
            try:
                backup_data = json.load(uploaded_backup)
                if st.button("Import Data"):
                    st.session_state.projects = backup_data.get("projects", {})
                    st.session_state.tasks = backup_data.get("tasks", {})
                    st.session_state.workspace_analytics = backup_data.get("analytics",
                                                                           {"sessions": 0, "ai_interactions": 0})
                    st.success("Workspace data imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing data: {e}")

        # Reset workspace
        st.markdown("---")
        st.subheader("🔄 Reset Workspace")
        st.warning("This will delete all projects, tasks, and analytics data.")

        if st.button("Reset All Data", type="secondary"):
            st.session_state.projects = {}
            st.session_state.tasks = {}
            st.session_state.workspace_analytics = {"sessions": 0, "ai_interactions": 0}
            st.session_state.ai_workspace_chat = []
            st.success("Workspace reset successfully!")
            st.rerun()

# Update session analytics
st.session_state.workspace_analytics["sessions"] = st.session_state.workspace_analytics.get("sessions", 0) + 1

# Tips and help
with st.expander("💡 Workspace Tips & Best Practices"):
    st.markdown("""
    **🚀 Maximizing Your AI Workspace:**

    **Project Management:**
    - Break large projects into smaller, manageable tasks
    - Use priority levels to focus on what matters most
    - Regular progress updates help AI provide better suggestions

    **AI Assistant Usage:**
    - Provide context about your projects for better help
    - Ask specific questions about challenges you're facing
    - Use AI suggestions to optimize your workflow

    **Analytics & Tracking:**
    - Monitor project progress regularly
    - Use analytics to identify bottlenecks
    - Track AI interaction patterns for insights

    **Automation Benefits:**
    - Set up automation rules to save time
    - Use AI to generate reports and summaries
    - Let AI help with resource allocation decisions

    **Collaboration Features:**
    - Export data to share with team members
    - Use AI assistant for meeting preparation
    - Generate project reports for stakeholders
    """)