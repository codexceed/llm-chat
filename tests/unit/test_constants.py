import hypothesis
import hypothesis.strategies

import chatbot.constants


def test_message_structure() -> None:
    """Test that chatbot.constants.Message can be created with required fields."""
    user_message: chatbot.constants.Message = {"role": "user", "content": "Hello"}
    assistant_message: chatbot.constants.Message = {"role": "assistant", "content": "Hi there"}

    assert user_message["role"] == "user"
    assert user_message["content"] == "Hello"
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == "Hi there"


@hypothesis.given(content=hypothesis.strategies.text())
def test_message_with_arbitrary_content(content: str) -> None:
    """Test chatbot.constants.Message creation with arbitrary content.

    Args:
        content: Arbitrary content string for the message.
    """
    message: chatbot.constants.Message = {"role": "user", "content": content}
    assert message["content"] == content


def test_file_types_values() -> None:
    """Test that chatbot.constants.FileTypes enum has expected values."""
    assert chatbot.constants.FileTypes.CODE.value == "code"
    assert chatbot.constants.FileTypes.MARKDOWN.value == "markdown"
    assert chatbot.constants.FileTypes.HTML.value == "html"
    assert chatbot.constants.FileTypes.TEXT.value == "text"
    assert chatbot.constants.FileTypes.UNKNOWN.value == "unknown"


def test_file_types_completeness() -> None:
    """Test that all file types are represented in mappings."""
    # All file types should have entries in chatbot.constants.FILE_EXTENSION_TYPE_MAPPING
    file_types_in_mapping = set(chatbot.constants.FILE_EXTENSION_TYPE_MAPPING.keys())
    expected_file_types = {
        chatbot.constants.FileTypes.CODE,
        chatbot.constants.FileTypes.MARKDOWN,
        chatbot.constants.FileTypes.HTML,
        chatbot.constants.FileTypes.TEXT,
    }

    # UNKNOWN is not expected to have extensions
    assert file_types_in_mapping == expected_file_types


def test_file_types_enum_members() -> None:
    """Test that chatbot.constants.FileTypes enum has exactly the expected members."""
    expected_members = {"CODE", "MARKDOWN", "HTML", "TEXT", "UNKNOWN"}
    actual_members = {member.name for member in chatbot.constants.FileTypes}
    assert actual_members == expected_members


def test_code_extensions() -> None:
    """Test that code file extensions are properly mapped."""
    code_extensions = chatbot.constants.FILE_EXTENSION_TYPE_MAPPING[chatbot.constants.FileTypes.CODE]

    # Test some common programming language extensions
    expected_code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".cs",
        ".php",
        ".rb",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".sql",
        ".r",
        ".m",
        ".pl",
    }

    assert expected_code_extensions.issubset(code_extensions)

    # All extensions should start with a dot
    assert all(ext.startswith(".") for ext in code_extensions)

    # All extensions should be lowercase
    assert all(ext.islower() for ext in code_extensions)


def test_markdown_extensions() -> None:
    """Test that markdown file extensions are properly mapped."""
    markdown_extensions = chatbot.constants.FILE_EXTENSION_TYPE_MAPPING[chatbot.constants.FileTypes.MARKDOWN]
    assert markdown_extensions == {".md"}


def test_html_extensions() -> None:
    """Test that HTML file extensions are properly mapped."""
    html_extensions = chatbot.constants.FILE_EXTENSION_TYPE_MAPPING[chatbot.constants.FileTypes.HTML]
    assert html_extensions == {".html", ".htm"}


def test_text_extensions() -> None:
    """Test that text file extensions are properly mapped."""
    text_extensions = chatbot.constants.FILE_EXTENSION_TYPE_MAPPING[chatbot.constants.FileTypes.TEXT]
    assert text_extensions == {".txt"}


def test_no_extension_overlap() -> None:
    """Test that file extensions don't overlap between types."""
    all_extensions: set[str] = set()

    for extensions in chatbot.constants.FILE_EXTENSION_TYPE_MAPPING.values():
        # Check that no extension appears in multiple file types
        overlap = all_extensions.intersection(extensions)
        assert len(overlap) == 0, f"Extension overlap found: {overlap}"
        all_extensions.update(extensions)


def test_extensions_format() -> None:
    """Test that all extensions follow the expected format."""
    for extensions in chatbot.constants.FILE_EXTENSION_TYPE_MAPPING.values():
        for ext in extensions:
            # All extensions should start with a dot
            assert ext.startswith("."), f"Extension {ext} should start with '.'"

            # Extensions should be lowercase
            assert ext.islower(), f"Extension {ext} should be lowercase"

            # Extensions should not be empty or just a dot
            assert len(ext) > 1, f"Extension {ext} should have content after '.'"


def test_common_language_mappings() -> None:
    """Test that common programming languages are properly mapped."""
    expected_mappings = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".cs": "c_sharp",
        ".php": "php",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".ps1": "powershell",
        ".sql": "sql",
        ".r": "r",
        ".m": "objective_c",
        ".pl": "perl",
    }

    for ext, expected_lang in expected_mappings.items():
        assert chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING[ext] == expected_lang


def test_all_extensions_in_code_type() -> None:
    """Test that all extensions in language mapping are also in code file type."""
    code_extensions = chatbot.constants.FILE_EXTENSION_TYPE_MAPPING[chatbot.constants.FileTypes.CODE]
    language_mapping_extensions = set(chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING.keys())

    # All extensions in language mapping should be in code extensions
    assert language_mapping_extensions.issubset(code_extensions), (
        f"Extensions in language mapping but not in code type: {language_mapping_extensions - code_extensions}"
    )


def test_extension_format_consistency() -> None:
    """Test that extensions in language mapping follow the same format as file type mapping."""
    for ext in chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING:
        # All extensions should start with a dot
        assert ext.startswith("."), f"Extension {ext} should start with '.'"

        # Extensions should be lowercase
        assert ext.islower(), f"Extension {ext} should be lowercase"

        # Extensions should not be empty or just a dot
        assert len(ext) > 1, f"Extension {ext} should have content after '.'"


def test_language_names_format() -> None:
    """Test that language names follow expected format."""
    for ext, language in chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING.items():
        # Language names should be strings
        assert isinstance(language, str), f"Language for {ext} should be string, got {type(language)}"

        # Language names should not be empty
        assert len(language) > 0, f"Language name for {ext} should not be empty"

        # Language names should be lowercase or snake_case
        assert language.islower() or "_" in language, f"Language name '{language}' should be lowercase or snake_case"


@hypothesis.given(random_ext=hypothesis.strategies.text())
def test_get_language_for_unknown_extension(random_ext: str) -> None:
    """Test behavior when looking up unknown extensions.

    Args:
        random_ext: Random file extension to test.
    """
    # Only test extensions that start with '.' and aren't in the mapping
    if random_ext.startswith(".") and random_ext not in chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING:
        result = chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING.get(random_ext, "unknown")
        assert result == "unknown"


def test_javascript_typescript_consistency() -> None:
    """Test that JavaScript and TypeScript extensions are properly handled."""
    # JavaScript extensions
    js_extensions = [".js", ".jsx"]
    for ext in js_extensions:
        assert chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING[ext] == "javascript"

    # TypeScript extensions
    ts_extensions = [".ts", ".tsx"]
    for ext in ts_extensions:
        assert chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING[ext] == "typescript"


def test_shell_script_consistency() -> None:
    """Test that shell script extensions map to bash."""
    shell_extensions = [".sh", ".bash", ".zsh"]
    for ext in shell_extensions:
        assert chatbot.constants.EXTENSION_TO_LANGUAGE_MAPPING[ext] == "bash"
