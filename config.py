"""
Configuration management for GeoAI RAG Assistant
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class AppConfig(BaseSettings):
    """Application configuration with environment variable support"""

    # LLM Configuration
    llm_provider: str = Field('mistral', env='LLM_PROVIDER')  # 'mistral' or 'openai'
    mistral_api_key: Optional[str] = Field(None, env='MISTRAL_API_KEY')
    mistral_model: str = Field('mistral-small-latest', env='MISTRAL_MODEL')
    # Available Mistral models: mistral-small-latest, mistral-medium-latest, mistral-large-latest, open-mistral-7b
    llm_temperature: float = Field(0.7, env='LLM_TEMPERATURE')

    # Embedding Configuration
    embedding_model: str = Field('all-MiniLM-L6-v2', env='EMBEDDING_MODEL')
    embedding_batch_size: int = Field(32, env='EMBEDDING_BATCH_SIZE')

    # Vector Store Configuration
    vector_store_type: str = Field('flat', env='VECTOR_STORE_TYPE')  # flat, ivf, hnsw
    vector_store_path: str = Field('./data/vector_store', env='VECTOR_STORE_PATH')

    # Retrieval Configuration
    default_top_k: int = Field(5, env='DEFAULT_TOP_K')
    max_top_k: int = Field(20, env='MAX_TOP_K')

    # Data Configuration
    data_cache_dir: str = Field('./data/cache', env='DATA_CACHE_DIR')
    enable_data_caching: bool = Field(True, env='ENABLE_DATA_CACHING')

    # Dataset Limits (for development/testing)
    earthquake_limit: int = Field(500, env='EARTHQUAKE_LIMIT')
    accident_limit: int = Field(2000, env='ACCIDENT_LIMIT')
    demographics_limit: int = Field(100, env='DEMOGRAPHICS_LIMIT')
    infrastructure_limit: int = Field(500, env='INFRASTRUCTURE_LIMIT')

    # API Configuration
    nrel_api_key: str = Field('DEMO_KEY', env='NREL_API_KEY')  # For EV charging stations

    # Map Configuration
    default_map_center: tuple = (37.0, -95.0)  # USA center
    default_map_zoom: int = 4

    # Logging
    log_level: str = Field('INFO', env='LOG_LEVEL')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get application configuration (singleton)"""
    global _config

    if _config is None:
        _config = AppConfig()

    return _config


def reload_config():
    """Reload configuration from environment"""
    global _config
    _config = AppConfig()
    return _config


# Dataset configurations
DATASET_CONFIG = {
    'earthquakes': {
        'name': 'USGS Earthquakes',
        'description': 'Recent earthquake events from USGS',
        'update_frequency': 'Real-time',
        'url': 'https://earthquake.usgs.gov/',
        'color': 'red',
        'icon': 'warning'
    },
    'traffic_accidents': {
        'name': 'NYC Motor Vehicle Collisions',
        'description': 'Traffic accident data from NYC Open Data',
        'update_frequency': 'Daily',
        'url': 'https://data.cityofnewyork.us/',
        'color': 'orange',
        'icon': 'exclamation-triangle'
    },
    'demographics': {
        'name': 'US Census Demographics',
        'description': 'County-level demographic information',
        'update_frequency': 'Annual',
        'url': 'https://www.census.gov/',
        'color': 'blue',
        'icon': 'users'
    },
    'infrastructure': {
        'name': 'Infrastructure Data',
        'description': 'EV charging stations and public facilities',
        'update_frequency': 'Monthly',
        'url': 'https://www.nrel.gov/',
        'color': 'green',
        'icon': 'building'
    }
}


# Example query templates
EXAMPLE_QUERIES = [
    {
        'question': 'Show recent earthquakes in California and populations at risk',
        'description': 'Combines earthquake and demographic data',
        'domains': ['earthquake', 'demographics']
    },
    {
        'question': 'Which neighborhoods in NYC have the highest accident rates near schools?',
        'description': 'Spatial analysis of accidents and schools',
        'domains': ['traffic_accident', 'infrastructure']
    },
    {
        'question': 'Where should new EV charging stations go in Boston?',
        'description': 'Infrastructure gap analysis',
        'domains': ['infrastructure', 'demographics']
    },
    {
        'question': 'Which counties in Florida are most vulnerable to both flooding and elderly population density?',
        'description': 'Multi-factor vulnerability assessment',
        'domains': ['demographics', 'disaster']
    },
    {
        'question': 'What are the strongest earthquakes in the last 30 days?',
        'description': 'Recent earthquake analysis',
        'domains': ['earthquake']
    },
    {
        'question': 'Show traffic accidents with fatalities in Manhattan',
        'description': 'Severe accident filtering',
        'domains': ['traffic_accident']
    }
]


# Prompt templates
SYSTEM_PROMPT = """You are a GeoAI assistant specializing in geospatial data analysis.
You help users understand patterns in earthquakes, traffic accidents, demographics, and infrastructure.

Your responses should:
1. Directly answer the user's question using the provided data
2. Highlight key insights and patterns
3. Include specific numbers, locations, and statistics when available
4. Mention data sources for credibility
5. Suggest actionable insights when appropriate
6. Be concise but informative (2-4 paragraphs)

If the data is insufficient, say so clearly and suggest what additional data would help.

Powered by Mistral AI."""


USER_PROMPT_TEMPLATE = """Based on the following geospatial data, please answer this question:

Question: {query}

Retrieved Data:
{context}

Please provide a clear, data-driven answer with specific details from the sources."""


# API endpoints and data sources
DATA_SOURCES = {
    'usgs_earthquake': 'https://earthquake.usgs.gov/fdsnws/event/1/query',
    'nyc_collisions': 'https://data.cityofnewyork.us/resource/h9gi-nx95.json',
    'nrel_ev_stations': 'https://developer.nrel.gov/api/alt-fuel-stations/v1.json',
    'census_api': 'https://api.census.gov/data',
}


if __name__ == "__main__":
    # Test configuration
    config = get_config()

    print("Configuration:")
    print(f"  LLM Provider: {config.llm_provider}")
    print(f"  Mistral Model: {config.mistral_model}")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  Vector Store: {config.vector_store_type}")
    print(f"  Default Top-K: {config.default_top_k}")
    print(f"  Data Cache: {config.data_cache_dir}")

    print("\nDatasets:")
    for key, info in DATASET_CONFIG.items():
        print(f"  {info['name']}: {info['description']}")

    print("\nExample Queries:")
    for i, query in enumerate(EXAMPLE_QUERIES[:3], 1):
        print(f"  {i}. {query['question']}")