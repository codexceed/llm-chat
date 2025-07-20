import streamlit as st

from chatbot import chat, constants, ui


def main() -> None:
    """Main function for the Streamlit chatbot app."""
    st.logo("assets/rand_logo.jpg")
    st.set_page_config(layout="wide")
    ui.render_sidebar()

    if chat_input := ui.render_chat_interface():
        prompt, uploaded_files = chat_input.text, chat_input.files
        if uploaded_files:
            chat.process_uploaded_files(uploaded_files)

        st.session_state.messages.append(constants.Message(role="user", content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            chat.generate_response(st.session_state.messages)


if __name__ == "__main__":
    main()
