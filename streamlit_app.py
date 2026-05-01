import os

import streamlit as st

from logging_config import configure_logging
from pipeline import run_finance_pipeline

configure_logging()

st.set_page_config(page_title="Finance Decision Assistant", page_icon=":bar_chart:", layout="centered")
st.title("Finance Decision Assistant")
st.caption("Chat with the assistant. Your query appears above the response, and the input is ready for the next prompt.")

# Sidebar: API key configuration
with st.sidebar:
    st.header("Configuration")

    provider = st.selectbox("LLM Provider", ["groq", "openai"], index=0)

    if provider == "groq":
        key_label, env_var = "Groq API Key", "GROQ_API_KEY"
        st.markdown("Get a free key at [console.groq.com](https://console.groq.com)")
    else:
        key_label, env_var = "OpenAI API Key", "OPENAI_API_KEY"
        st.markdown("Get a key at [platform.openai.com](https://platform.openai.com)")

    api_key_input = st.text_input(key_label, type="password", placeholder="Paste your key here")

    if api_key_input:
        os.environ[env_var] = api_key_input
        os.environ["LLM_PROVIDER"] = provider
        st.success("API key set for this session.")

api_key_configured = bool(api_key_input) or bool(os.getenv("GROQ_API_KEY")) or bool(os.getenv("OPENAI_API_KEY"))

# Chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("is_error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("debug"):
            debug = message["debug"]
            with st.expander("Debug info"):
                st.write("Route:", debug.get("route"))
                st.write("Risk:", debug.get("risk"))
                st.write("Reasons:", debug.get("reasons", []))
                st.write("Full output:", debug.get("full_output", {}))


for message in st.session_state.messages:
    render_message(message)

# Chat input
if not api_key_configured:
    st.info("Enter your API key in the sidebar to start chatting.")
    st.stop()

prompt = st.chat_input("Ask a finance question...")

if prompt:
    clean_query = prompt.strip()
    if not clean_query:
        st.warning("Please enter a query.")
    else:
        user_message = {"role": "user", "content": clean_query}
        st.session_state.messages.append(user_message)
        render_message(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query..."):
                try:
                    state = run_finance_pipeline(clean_query)
                    assistant_text = state.get("final_response", "")
                    if not assistant_text:
                        raise RuntimeError("Pipeline completed without a final response.")

                    st.markdown(assistant_text)

                    profile = state.get("profile")
                    risk = getattr(profile, "risk", None)

                    debug_data = {
                        "route": state.get("route"),
                        "risk": risk,
                        "reasons": state.get("reasons", []),
                        "full_output": state,
                    }
                    with st.expander("Debug info"):
                        st.write("Route:", debug_data["route"])
                        st.write("Risk:", debug_data["risk"])
                        st.write("Reasons:", debug_data["reasons"])
                        st.write("Full output:", debug_data["full_output"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_text,
                            "debug": debug_data,
                        }
                    )
                except Exception as exc:
                    error_message = f"Error analyzing query: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message, "is_error": True}
                    )
