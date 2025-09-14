"""Entry point for the chatbot CLI."""

import os
import sys

import streamlit.web.cli as stcli


def main() -> None:
    """Entry point for the chatbot CLI."""
    # Get the path to the Streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    # Run the Streamlit app with all passed arguments
    sys.argv = ["streamlit", "run", app_path, *sys.argv[1:]]
    stcli.main()


if __name__ == "__main__":
    main()
