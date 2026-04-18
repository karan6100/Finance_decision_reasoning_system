import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv()

class LLMFactory:
    _configs: Optional[Dict[str, Any]] = None

    @classmethod
    def _load_configs(cls) -> Dict[str, Any]:
        """Load model configurations from YAML file"""
        if cls._configs is None:
            config_path = Path(__file__).parent / "model_configs.yaml"
            try:
                with open(config_path, 'r') as f:
                    cls._configs = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {e}")

        return cls._configs

    @staticmethod
    def get_llm(provider: str = 'groq', task_type: str = 'basic', model_name: Optional[str] = None):
        """
        Get an LLM instance with appropriate configuration.

        Args:
            provider: 'groq' or 'openai'
            task_type: 'basic', 'reasoning', or 'complex'
            model_name: Override model name (optional)
        """
        configs = LLMFactory._load_configs()

        if model_name:
            # Use custom model with default temperature
            config = {'model': model_name, 'temperature': 0.0}
        else:
            # Get config from YAML based on task_type and provider
            task_configs = configs.get(task_type, configs.get('basic', {}))
            config = task_configs.get(provider, {})

            if not config:
                raise ValueError(f"No configuration found for provider '{provider}' and task_type '{task_type}'")

        if provider == 'groq':
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set in environment!")

            return ChatGroq(
                model=config['model'],
                temperature=config['temperature'],
                api_key=api_key
            )

        elif provider == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment!")

            return ChatOpenAI(
                model=config['model'],
                temperature=config['temperature'],
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_available_configs() -> Dict[str, Any]:
        """Get all available configurations for inspection"""
        return LLMFactory._load_configs()

    @staticmethod
    def reload_configs():
        """Force reload configurations from file (useful for development)"""
        LLMFactory._configs = None
        return LLMFactory._load_configs()

