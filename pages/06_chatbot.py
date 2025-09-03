import streamlit as st
from utils.gemini_client import chat_with_context, initialize_gemini_client

# Initialize Gemini client
initialize_gemini_client()

st.title("💬 AI Chatbot")
st.markdown("Interactive conversation with Google Gemini AI")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_context" not in st.session_state:
    st.session_state.chat_context = ""

# Sidebar for chat settings
with st.sidebar:
    st.header("🛠️ Chat Settings")

    # Chat mode
    chat_mode = st.selectbox(
        "Chat Mode:",
        ["General Assistant", "Code Helper", "Writing Assistant", "Research Helper", "Creative Partner"]
    )

    # Context setting
    st.subheader("Context")
    context_input = st.text_area(
        "Set context for the conversation:",
        value=st.session_state.chat_context,
        height=100,
        help="Provide background information to help the AI understand your needs better"
    )

    if st.button("Update Context"):
        st.session_state.chat_context = context_input
        st.success("Context updated!")

    # Quick context templates
    st.subheader("Quick Context Templates")
    templates = {
        "Student": "I am a student looking for help with learning and understanding concepts.",
        "Developer": "I am a software developer working on programming projects.",
        "Writer": "I am a writer looking for creative assistance and editing help.",
        "Business": "I am working on business-related tasks and need professional assistance.",
        "Researcher": "I am conducting research and need help with analysis and information gathering."
    }

    template_choice = st.selectbox("Select template:", ["None"] + list(templates.keys()))
    if st.button("Apply Template") and template_choice != "None":
        st.session_state.chat_context = templates[template_choice]
        st.success(f"Applied {template_choice} context!")

    # Clear chat
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# Chat mode specific instructions
mode_instructions = {
    "General Assistant": "I'm here to help with general questions and tasks.",
    "Code Helper": "I'll help you with programming, debugging, and code explanations.",
    "Writing Assistant": "I'll assist with writing, editing, and creative content.",
    "Research Helper": "I'll help you research topics and analyze information.",
    "Creative Partner": "Let's collaborate on creative projects and brainstorming!"
}

st.info(f"**{chat_mode}**: {mode_instructions[chat_mode]}")

# Display chat history
chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create context for the chat
            full_context = ""
            if st.session_state.chat_context:
                full_context += f"Context: {st.session_state.chat_context}\n"

            full_context += f"Chat Mode: {chat_mode}\n"

            # Add recent chat history for context
            if len(st.session_state.chat_history) > 1:
                recent_history = st.session_state.chat_history[-6:-1]  # Last 5 messages (excluding current)
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
                full_context += f"Recent conversation:\n{history_text}\n"

            response = chat_with_context(user_input, full_context)
            st.write(response)

            # Add AI response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Quick action buttons
if st.session_state.chat_history:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📝 Summarize Chat"):
            with st.spinner("Summarizing conversation..."):
                chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                summary_context = f"Summarize this conversation:\n{chat_text}"
                summary = chat_with_context("Please provide a concise summary of our conversation.", summary_context)
                st.write("**Conversation Summary:**")
                st.write(summary)

    with col2:
        if st.button("🔑 Key Points"):
            with st.spinner("Extracting key points..."):
                chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                key_points_context = f"Extract key points from this conversation:\n{chat_text}"
                key_points = chat_with_context(
                    "Please list the key points and important information from our conversation.", key_points_context)
                st.write("**Key Points:**")
                st.write(key_points)

    with col3:
        if st.button("❓ Follow-up Questions"):
            with st.spinner("Generating questions..."):
                chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                questions_context = f"Based on this conversation:\n{chat_text}"
                questions = chat_with_context(
                    "Suggest 3-5 relevant follow-up questions I could ask to continue this conversation.",
                    questions_context)
                st.write("**Suggested Follow-up Questions:**")
                st.write(questions)

    with col4:
        if st.button("📋 Export Chat"):
            chat_export = ""
            for msg in st.session_state.chat_history:
                chat_export += f"**{msg['role'].title()}:** {msg['content']}\n\n"

            st.download_button(
                label="Download Chat",
                data=chat_export,
                file_name="chat_history.txt",
                mime="text/plain"
            )

# Suggested prompts for new users
if not st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💡 Suggested Conversation Starters")

    suggestions = {
        "General Assistant": [
            "Help me plan my day",
            "Explain a complex topic in simple terms",
            "Give me some productivity tips"
        ],
        "Code Helper": [
            "Help me debug this code",
            "Explain this programming concept",
            "Review my code for improvements"
        ],
        "Writing Assistant": [
            "Help me improve this paragraph",
            "Brainstorm ideas for my article",
            "Check my writing for grammar and style"
        ],
        "Research Helper": [
            "Help me research this topic",
            "Analyze these findings",
            "Create a research outline"
        ],
        "Creative Partner": [
            "Let's brainstorm creative ideas",
            "Help me with a creative project",
            "Generate some creative prompts"
        ]
    }

    current_suggestions = suggestions.get(chat_mode, suggestions["General Assistant"])

    col1, col2, col3 = st.columns(3)

    for i, suggestion in enumerate(current_suggestions):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                # Simulate user input
                st.session_state.chat_history.append({"role": "user", "content": suggestion})

                # Generate response
                full_context = ""
                if st.session_state.chat_context:
                    full_context += f"Context: {st.session_state.chat_context}\n"
                full_context += f"Chat Mode: {chat_mode}\n"

                response = chat_with_context(suggestion, full_context)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

# Tips and help
with st.expander("💡 Tips for Better Conversations"):
    st.markdown("""
    **Getting the best responses:**
    - Be specific and clear in your questions
    - Provide context when needed
    - Use the context setting for ongoing projects
    - Try different chat modes for specialized help

    **Chat Modes:**
    - **General Assistant**: Best for everyday questions and tasks
    - **Code Helper**: Optimized for programming and technical help
    - **Writing Assistant**: Great for content creation and editing
    - **Research Helper**: Ideal for analysis and information gathering
    - **Creative Partner**: Perfect for brainstorming and creative projects

    **Pro Tips:**
    - Set context at the beginning of your session
    - Use follow-up questions to dive deeper
    - Export important conversations for later reference
    - Clear chat history when switching to a new topic
    """)
