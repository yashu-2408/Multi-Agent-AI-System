import streamlit as st
from research_agents import ResearchCrew

st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="centered")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Multi-Agent Research Assistant")
st.markdown("Generate comprehensive research reports using a team of specialized AI agents.")

with st.sidebar:
    st.header("Settings")

    st.caption("Keys are never stored — you'll need to re-enter your key each session.")

    groq_api_key = st.text_input("Groq API Key", value="", type="password")

    st.info("Built with CrewAI, Groq (Llama 3.3 70B), and Streamlit.")

topic = st.text_input("What do you want to research?", placeholder="e.g. The impact of Generative AI on Software Engineering")
instructions = st.text_area("Specific Instructions (Optional)", placeholder="Focus on the next 5 years...")

if st.button("Generate Report"):
    if not topic:
        st.warning("Please enter a topic to start.")
    elif not groq_api_key:
        st.error("Please provide a Groq API Key in the sidebar.")
    else:
        try:
            with st.status("🤖 AI Agents are working...", expanded=True) as status:
                st.write("📋 Planning and Gathering Data...")
                crew = ResearchCrew(topic, instructions, groq_api_key)
                result = crew.run()
                status.update(label="✅ Research Complete!", state="complete", expanded=False)

            st.success("Successfully generated report!")
            st.markdown("---")
            st.markdown(str(result))

            st.download_button(
                label="📥 Download Report as Markdown",
                data=str(result),
                file_name=f"research_{topic.replace(' ', '_').lower()}.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.caption("If this is a rate-limit error, wait a moment and try again.")

with st.expander("💡 How this works"):
    st.write("""
    This app demonstrates a **Multi-Agent Orchestration** pattern:
    1. **Separation of Concerns**: Each agent (Planner, Researcher, Writer) has a specific role and backstory.
    2. **Sequential Workflow**: Tasks are passed from one agent to the next, ensuring structured output.
    3. **Tool Integration**: The Researcher uses DuckDuckGo Search to pull live web data.
    4. **Groq + Llama 3.3 70B**: Ultra-fast inference with a generous free tier.
    """)
