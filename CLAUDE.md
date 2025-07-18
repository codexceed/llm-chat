# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning project that involves the research, development and deployment of state-of-the-art AI tools via a chatbot application. Long-term objectives include:

- Optimizing an LLM-powered chatbot stack which includes highly performant inference backends and incredibly helpful frontend UI/UX.
- Continuously improve scalability and responsiveness of the chatbot.
- Integrate the latest in AI technologies into the chatbot.

## Architecture

The codebase is organized into a single `chatbot` package with the following structure:

- `app.py` - Main Streamlit application entry point
- `cli.py` - Command-line interface launcher  
- `chat.py` - Core chat functionality and OpenAI API integration
- `ui.py` - Streamlit UI components and chat interface
- `config.py` - Pydantic settings with `.env` file support
- `types.py` - Type definitions for messages and data structures

The application follows a modular design where UI rendering, chat logic, and configuration are separated into distinct modules.

## Development Commands

### Setup
```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Set up complete development environment (includes pre-commit hooks)
make dev
```

### Code Quality
```bash
# Run all linting and formatting
make lint-all

# Individual tools
make format      # Format with ruff
make lint        # Check with ruff  
make type-check  # Type check with mypy
make security    # Security scan with bandit
make lint-fix    # Auto-fix linting issues

# Quick validation (no formatting changes)
make check
```

### Testing
```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Unit tests only
make test-unit
```

### CI Pipeline
```bash
# Full CI workflow (used in GitHub Actions)
make ci
```

## Configuration

The application uses Pydantic settings loaded from `.env` file:

- `OPENAI_API_BASE` - API endpoint (defaults to localhost:8000/v1)
- `MODEL_NAME` - Model identifier for OpenAI API calls
- `TEMPERATURE`, `MAX_TOKENS`, `SEED` - Model parameters
- `UPLOAD_DIR` - Directory for file uploads
- `PAGE_TITLE` - Streamlit app title
- `HOST`, `PORT` - Server configuration

## Docker Infrastructure

The project includes Docker Compose configuration for running VLLM model servers:

- `vllm-qwen-2.5-coder-14b-instruct-awq` - 14B model service
- `vllm-qwen-2.5-coder-7b-instruct-awq` - 7B model service (default)

Both services expose OpenAI-compatible APIs on port 8000.

## Pre-commit Hooks

The project uses pre-commit hooks for code quality:
- Ruff (linting and formatting)
- MyPy (type checking) 
- Bandit (security scanning)
- Commitizen (conventional commit messages)
- nbQA (Jupyter notebook support)

## Application Entry Points

- `chatbot` - CLI command to launch the Streamlit app
- Direct execution: `python -m chatbot.app` or `streamlit run chatbot/app.py`

## Coding Style & Conventions

### Google Python Style Guide

- **Python Language Rules**
  - For code linting:
    - Use ruff for formatting.
    - mypy or pyright for static analysis and type checking.
  - Use imports for packages and modules only, not individual classes or functions
  - Import each module using full pathname location
  - Use exceptions for exceptional conditions, not control flow
  - Avoid mutable global state
  - Nested functions/classes are fine when closing over local values@chat
  - Use list/dict/set comprehensions for simple cases only
  - Use default iterators and operators for types that support them
  - Use generators as needed
  - Use lambda functions only for one-liners
  - Conditional expressions (x if y else z) OK for simple cases
  - Default argument values should never be mutable objects
  - Properties should be used when simple computations are needed
  - Use implicit False tests (if foo: instead of if foo != []:)
  - Use decorator syntax judiciously
  - Avoid fancy language features (metaclasses, bytecode access, etc.)
  - Use from __future__ imports for Python compatibility
  - Use type annotations where helpful

- **Python Style Rules**
  - No semicolons
  - Maximum line length of 80 characters
  - Use parentheses sparingly
  - Indent code blocks with 4 spaces (never tabs)
  - Use trailing commas in sequences only when closing bracket is on another line
  - Use blank lines sparingly (two between top-level definitions)
  - No whitespace inside parentheses/brackets/braces
  - Use whitespace after commas/semicolons/colons
  - No trailing whitespace
  - Surround binary operators with single space on each side
  - Follow standard typographic rules for spaces around punctuation
  - Shebang line: #!/usr/bin/env python3 (if needed)

- **Docstrings**
  - Use """triple double quotes""" for docstrings
  - One-line docstring should be on one line
  - Multi-line docstrings: summary line, blank line, details
  - Modules should have docstrings describing contents and usage
  - Functions/methods should have docstrings with clear Args:/Returns:/Raises: sections
  - Class docstrings should include Attributes: section for public attributes

- **String Formatting**
  - Use f-strings, %-formatting, or str.format()
  - Avoid + operator for string concatenation in loops

- **Files and I/O**
  - Always use with statement for file and similar resource handling

- **TODO Comments**
  - Format: # TODO: link_to_issue - explanation
  - Must include context (ideally a bug reference) and explanation

- **Imports**
  - Group imports: __future__, standard library, third-party, application imports
  - Sort imports alphabetically within each group
  - Use absolute imports

- **Naming**
  - `module_name`, `package_name`
  - `ClassName`, `ExceptionName`
  - `function_name`, `method_name`, `instance_var_name`, `function_parameter_name`, `local_var_name`
  - `GLOBAL_CONSTANT_NAME`
  - Protected members start with single underscore `_`
  - Private instance variables use double underscore `__` (rarely needed)

- **Main Function**
  - Use `if __name__ == '__main__'` guard with main() function
  - Prefer small, focused functions (not strict limit, but review if over 40 lines)

- **Type Annotations**
  - Follow PEP 484 typing rules
  - Use type annotations for function signatures and variables
  - Add type annotations to APIs and complex code
  - Follow specific line breaking conventions for type annotations
  - Use forward declarations when necessary
  - Avoid conditional imports except when absolutely necessary
  - Use proper type aliases for complex types
  - Prefer abstract container types over concrete ones

- **Comments**
  - Comments should be complete sentences with proper punctuation
  - Use block comments for complex logic explanations
  - Inline comments should be separated by at least two spaces from code

### SOLID Design Principles - Coding Assistant Guidelines

When generating, reviewing, or modifying code, follow these guidelines to ensure adherence to SOLID principles:

#### 1. Single Responsibility Principle (SRP)

- Each class must have only one reason to change.
- Limit class scope to a single functional area or abstraction level.
- When a class exceeds 100-150 lines, consider if it has multiple responsibilities.
- Separate cross-cutting concerns (logging, validation, error handling) from business logic.
- Create dedicated classes for distinct operations like data access, business rules, and UI.
- Method names should clearly indicate their singular purpose.
- If a method description requires "and" or "or", it likely violates SRP.
- Prioritize composition over inheritance when combining behaviors.

#### 2. Open/Closed Principle (OCP)

- Design classes to be extended without modification.
- Use abstract classes and interfaces to define stable contracts.
- Implement extension points for anticipated variations.
- Favor strategy patterns over conditional logic.
- Use configuration and dependency injection to support behavior changes.
- Avoid switch/if-else chains based on type checking.
- Provide hooks for customization in frameworks and libraries.
- Design with polymorphism as the primary mechanism for extending functionality.

#### 3. Liskov Substitution Principle (LSP)

- Ensure derived classes are fully substitutable for their base classes.
- Maintain all invariants of the base class in derived classes.
- Never throw exceptions from methods that don't specify them in base classes.
- Don't strengthen preconditions in subclasses.
- Don't weaken postconditions in subclasses.
- Never override methods with implementations that do nothing or throw exceptions.
- Avoid type checking or downcasting, which may indicate LSP violations.
- Prefer composition over inheritance when complete substitutability can't be achieved.

#### 4. Interface Segregation Principle (ISP)

- Create focused, minimal interfaces with cohesive methods.
- Split large interfaces into smaller, more specific ones.
- Design interfaces around client needs, not implementation convenience.
- Avoid "fat" interfaces that force clients to depend on methods they don't use.
- Use role interfaces that represent behaviors rather than object types.
- Implement multiple small interfaces rather than a single general-purpose one.
- Consider interface composition to build up complex behaviors.
- Remove any methods from interfaces that are only used by a subset of implementing classes.

#### 5. Dependency Inversion Principle (DIP)

- High-level modules should depend on abstractions, not details.
- Make all dependencies explicit, ideally through constructor parameters.
- Use dependency injection to provide implementations.
- Program to interfaces, not concrete classes.
- Place abstractions in a separate package/namespace from implementations.
- Avoid direct instantiation of service classes with 'new' in business logic.
- Create abstraction boundaries at architectural layer transitions.
- Define interfaces owned by the client, not the implementation.

#### Implementation Guidelines

- When starting a new class, explicitly identify its single responsibility.
- Document extension points and expected subclassing behavior.
- Write interface contracts with clear expectations and invariants.
- Question any class that depends on many concrete implementations.
- Use factories, dependency injection, or service locators to manage dependencies.
- Review inheritance hierarchies to ensure LSP compliance.
- Regularly refactor toward SOLID, especially when extending functionality.
- Use design patterns (Strategy, Decorator, Factory, Observer, etc.) to facilitate SOLID adherence.

#### Warning Signs

- God classes that do "everything"
- Methods with boolean parameters that radically change behavior
- Deep inheritance hierarchies
- Classes that need to know about implementation details of their dependencies
- Circular dependencies between modules
- High coupling between unrelated components
- Classes that grow rapidly in size with new features
- Methods with many parameters
