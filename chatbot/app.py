import streamlit as st

from chatbot.chat import generate_response, process_uploaded_file
from chatbot.ui import render_chat_interface, render_sidebar


def main() -> None:
    """Main function for the Streamlit chatbot app."""
    st.set_page_config(layout="wide")
    uploaded_file = render_sidebar()

    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)

    if prompt := render_chat_interface():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        generate_response()


if __name__ == "__main__":
    main()
