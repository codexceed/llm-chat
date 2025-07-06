from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_base: str = "http://localhost:8000/v1"
    openai_api_key: str = "not-needed"
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
    temperature: float = 0.7
    max_tokens: int = 2000
    repetition_penalty: float = 1.1
    upload_dir: str = "uploads"
    page_title: str = "My Custom Chatbot"
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
