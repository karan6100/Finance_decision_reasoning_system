import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Finance Decision Assistant", page_icon=":bar_chart:", layout="centered")
st.title("Finance Decision Assistant")
st.caption("Chat with the assistant. Your query appears above the response, and the input is ready for the next prompt.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(message: dict) -> None:
    """Render one stored chat message."""
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
                    response = requests.post(
                        f"{BACKEND_URL}/analyze",
                        json={"query": clean_query},
                        timeout=200,
                    )

                    if response.ok:
                        data = response.json()
                        assistant_text = data.get("response", "No response returned.")
                        st.markdown(assistant_text)

                        debug_data = {
                            "route": data.get("route"),
                            "risk": data.get("risk"),
                            "reasons": data.get("reasons", []),
                            "full_output": data,
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
                    else:
                        error_message = f"API error {response.status_code}: {response.text}"
                        st.error(error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_message, "is_error": True}
                        )

                except requests.RequestException as exc:
                    error_message = f"Cannot connect to backend API at {BACKEND_URL}. Error: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message, "is_error": True}
                    )
