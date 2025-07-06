import streamlit as st
from chatbot.ui import render_sidebar, render_chat_interface
from chatbot.chat import process_uploaded_file, generate_response


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
