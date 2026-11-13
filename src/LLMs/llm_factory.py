import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from observability import usage_callback  # per-node token accounting


load_dotenv()

# One shared limiter for all LLM instances. 0.5 req/s means at most one call every 2s.
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0.5"))
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=LLM_MIN_INTERVAL_S,
    check_every_n_seconds=0.1, 
    max_bucket_size=1,   # no bursts: calls are strictly spaced
)

# The limiter above only spaces REQUESTS; Groq also enforces tokens/min per model, which
# still yields 429s. The provider SDK retries those itself, honouring the server's
# Retry-After header, so give it enough attempts to ride out a TPM window (default is 2).
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))

# Optional YAML keys forwarded to the client only when set (see model_configs.yaml).
_OPTIONAL_PARAMS = ("max_tokens", "reasoning_effort")


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

        optional = {k: config[k] for k in _OPTIONAL_PARAMS if config.get(k) is not None}

        if provider == 'groq':
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set in environment!")

            return ChatGroq(
                model=config['model'],
                temperature=config['temperature'],
                api_key=api_key,
                rate_limiter=_rate_limiter,
                max_retries=LLM_MAX_RETRIES,
                callbacks=[usage_callback],
                **optional,
            )

        elif provider == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment!")

            return ChatOpenAI(
                model=config['model'],
                temperature=config['temperature'],
                api_key=api_key,
                rate_limiter=_rate_limiter,
                max_retries=LLM_MAX_RETRIES,
                callbacks=[usage_callback],
                **optional,
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

