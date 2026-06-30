# 🚀 Multi-Agent Research Assistant

A lightweight, high-impact AI Research Assistant built with **Streamlit**, **CrewAI**, and **Groq + Llama 3.3 70B**. Designed to be easily hosted on **Streamlit Cloud** and explained during technical interviews.

## 🏗️ How it Works

```text
User Topic -> [ Planner Agent ] -> [ Researcher Agent ] -> [ Writer Agent ] -> Final Report
```

1.  **Planner**: Breaks the topic into 3-5 key research objectives.
2.  **Researcher**: Searches the live web using DuckDuckGo to gather data.
3.  **Writer**: Synthesizes all findings into a structured Markdown report/PDF.

## 🌟 Key Features

- **Team-Based AI**: Uses specialized agents instead of a single prompt.
- **Live Search**: Integrated web search for up-to-date data.
- **Export Ready**: Download your generated research as professional Markdown(PDF Download feature will be added soon).
- **Cloud Native**: Optimized for 1-click deployment on Streamlit Cloud.

## 🌐 Deployment (Streamlit Community Cloud)

This app is optimized for [Streamlit Community Cloud](https://share.streamlit.io/):

1. **GitHub**: Push this project to a public GitHub repository.
2. **Deploy**: Sign in to [Streamlit Cloud](https://share.streamlit.io/) and click "New app".
3. **Configure**: Select your repository, branch, and `app.py` as the main file.
4. **Launch**: Click "Deploy" and your multi-agent assistant will be live!

## 🛠️ Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📄 Project Highlights

- **Multi-Agent Orchestration**: Demonstrated ability to manage complex LLM workflows using CrewAI.
- **Effective Prompt Engineering**: Developed role-based agents with specific backstories and goals.
- **Rapid Prototyping**: Built a full-stack AI application with a clean, user-focused UI.
