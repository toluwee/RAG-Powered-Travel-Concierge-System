from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Intelligent Travel Concierge"
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./travel.db")
    
    # External API Settings
    EXPEDIA_API_KEY: Optional[str] = os.getenv("EXPEDIA_API_KEY", "")
    WEATHER_API_KEY: Optional[str] = os.getenv("WEATHER_API_KEY", "")
    
    # Model Settings
    MODEL_NAME: str = "gpt-4"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # RAG Settings
    VECTOR_DB_PATH: str = "data/vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        case_sensitive = True

settings = Settings() 