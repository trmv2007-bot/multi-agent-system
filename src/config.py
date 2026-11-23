"""Configuration settings for the multi-agent system."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thinking power: how hard the model is asked to work on each call.
# reasoning_effort goes to the gateway as a top-level field, max_tokens caps the
# reply. "off" sends no effort hint at all, for gateways that reject it.
THINKING_LEVELS: dict[str, tuple[str | None, int]] = {
    "off": (None, 2048),
    "low": ("low", 2048),
    "medium": ("medium", 4096),
    "high": ("high", 8192),
    "max": ("high", 16384),
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Both files are read whatever the working directory is: the project's .env
    # wins, and the user's home .env supplies provider keys the project does not
    # define. extra="ignore" lets a machine carry keys this project never uses
    # (MISTRAL_API_KEY, and so on) without the app refusing to start.
    model_config = SettingsConfigDict(
        env_file=(str(Path.home() / ".env"), str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # KeyPool Gateway (all AI calls route through here)
    keypool_base_url: str = Field(
        default="http://127.0.0.1:8787",
        description="KeyPool gateway base URL",
    )
    keypool_api_key: str | None = Field(
        default=None,
        description="KeyPool API key (set KEYPOOL_API_KEY in .env)",
    )
    keypool_model: str = Field(
        default="pooled-model",
        description="Model ID for KeyPool gateway",
    )
    keypool_protocol: str = Field(
        default="anthropic_messages",
        description="Protocol: anthropic_messages or openai",
    )

    # Local LLM Providers (Ollama, LM Studio, LocalAI, etc.)
    local_llm_base_url: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="Local LLM base URL (Ollama, LM Studio, LocalAI)",
    )
    local_llm_api_key: str | None = Field(
        default="local",
        description="Local LLM API key (usually not needed)",
    )
    local_llm_model: str = Field(
        default="llama3.2",
        description="Local model ID (e.g., llama3.2, mistral, codellama)",
    )

    # Free fallback providers (OpenAI-compatible)
    openrouter_api_key: str | None = Field(
        default=None,
        description="OpenRouter API key (free models: :free suffix)",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL",
    )
    openrouter_free_model: str = Field(
        default="openrouter/free",
        description="OpenRouter free model router (picks a free model)",
    )

    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key (free tier, no card needed)",
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible base URL",
    )
    groq_free_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq free model",
    )

    google_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key (free tier on Flash models)",
    )
    google_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Gemini OpenAI-compatible base URL",
    )
    google_free_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini free model",
    )

    cloudflare_api_key: str | None = Field(
        default=None,
        description="Cloudflare Workers AI API token",
    )
    cloudflare_account_id: str | None = Field(
        default=None,
        description="Cloudflare account ID",
    )
    cloudflare_free_model: str = Field(
        default="@hf/meta-llama/llama-3.2-3b-instruct",
        description="Cloudflare free model",
    )

    # Provider selection
    llm_provider: str = Field(
        default="keypool",
        description="Primary provider: keypool, local, openrouter, groq, google, cloudflare",
    )
    enable_free_fallback: bool = Field(
        default=False,
        description=(
            "Route every call through LLM_PROVIDER only; set true to add the other "
            "configured providers as fallbacks"
        ),
    )
    max_fallback_retries: int = Field(
        default=4,
        description="Attempts per LLM call before switching provider, then giving up",
    )
    llm_retry_backoff: float = Field(
        default=3.0,
        description="Base seconds for exponential backoff between retries",
    )
    thinking_level: str = Field(
        default="medium",
        description="Thinking power per call: off, low, medium, high or max",
    )
    whisper_model: str = Field(
        default="medium",
        description="Local speech-to-text model: tiny, base, small, medium, large-v3",
    )
    whisper_language: str = Field(
        default="en",
        description="Language for transcription, or empty to detect it",
    )
    whisper_device: str = Field(
        default="auto", description="auto, cpu or cuda for the local speech model"
    )
    whisper_compute: str = Field(
        default="",
        description="CTranslate2 compute type; empty picks float16 on GPU, int8 on CPU",
    )
    computer_use: bool = Field(
        default=False,
        description="Let agents look at the desktop and control the mouse and keyboard",
    )
    computer_max_actions: int = Field(
        default=25,
        description="Input actions a single turn may perform when computer use is on",
    )
    cursor_overlay: bool = Field(
        default=False,
        description="Show the live cursor overlay on every run, without needing --cursor",
    )
    cursor_settle_ms: int = Field(
        default=150,
        description="Pause before a computer action so the cursor visibly arrives; 0 disables it",
    )
    cursor_glide_ms: int = Field(
        default=200,
        description="How long the on-screen agent cursor takes to travel to its target",
    )
    tool_retries: int = Field(
        default=2,
        description="Attempts per tool call when the network fails or the source errors",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        description=(
            "Hard cap on one model call. Without it a stalled gateway connection "
            "waits forever and the run looks alive but never returns."
        ),
    )

    # ── Other providers (paid fallback only) ──────────────────────────
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model")
    anthropic_model: str = Field(
        default="claude-3.5-sonnet-20241022", description="Anthropic model"
    )

    # GitHub (optional: raises the unauthenticated search rate limit)
    github_token: str | None = Field(
        default=None,
        description="GitHub token for the MCP search_github tool",
    )

    # Database (leave unset to use the in-memory checkpointer)
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL connection string for the durable checkpointer",
    )
    checkpoint_db: str | None = Field(
        default="multi-agent.sqlite",
        description=(
            "SQLite file holding durable run state; "
            "set CHECKPOINT_DB= (empty) to keep runs in memory only"
        ),
    )

    # LangSmith
    langsmith_api_key: str | None = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(
        default="multi-agent-system", description="LangSmith project name"
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", description="LangSmith endpoint"
    )

    # OTel
    otel_enabled: bool = Field(
        default=False,
        description="Export traces and metrics to the OTLP collector at OTEL_ENDPOINT",
    )
    otel_endpoint: str = Field(
        default="http://localhost:4317", description="OTel collector endpoint"
    )
    otel_service_name: str = Field(default="multi-agent-system", description="OTel service name")

    # Dashboard
    dashboard_host: str = Field(default="127.0.0.1", description="Dashboard bind host")
    dashboard_port: int = Field(default=8765, description="Dashboard port")

    # MCP Server
    mcp_server_host: str = Field(
        default="127.0.0.1",
        description="MCP server host; anything beyond loopback needs SERVER_AUTH_TOKEN",
    )
    mcp_server_port: int = Field(default=8001, description="MCP server port")

    # A2A Server
    a2a_server_host: str = Field(
        default="127.0.0.1",
        description="A2A server host; anything beyond loopback needs SERVER_AUTH_TOKEN",
    )
    a2a_server_port: int = Field(default=8002, description="A2A server port")

    # Shared by both HTTP servers: set it to let them bind beyond loopback, and
    # every request then has to carry "Authorization: Bearer <token>".
    server_auth_token: str | None = Field(
        default=None,
        description="Bearer token the MCP and A2A HTTP servers require on every request",
    )

    # Application
    log_level: str = Field(default="INFO", description="Logging level")
    max_parallel_agents: int = Field(default=5, description="Maximum parallel agents")
    thread_id: str = Field(default="default-session", description="Default thread ID")


# Global settings instance
settings = Settings()

HTTP_OK = 200


def models_url(provider: str, base_url: str) -> str:
    """Model-listing endpoint for a provider base URL, tolerating a trailing /v1."""
    base = base_url.rstrip("/")
    if provider == "keypool" and not base.endswith("/v1"):
        return f"{base}/v1/models"
    return f"{base}/models"


def get_llm(provider: str | None = None) -> ChatOpenAI:
    """
    Get an LLM instance.

    Args:
        provider: Specific provider name, or None for the configured primary.

    Returns:
        A ChatOpenAI instance configured for the selected provider.

    Example:
        ```python
        # Use primary provider (KeyPool by default)
        llm = get_llm()

        # Use a specific provider
        llm = get_llm("groq")

        # Failover to next provider
        llm = router.next_provider()
        ```
    """
    return router.get_llm(provider)


def get_router() -> "FreeLLMRouter":
    """Get the global router instance."""
    return router


# ── Free LLM Router ──────────────────────────────────────────────────
@dataclass(frozen=True)
class Provider:
    """One link in the failover chain."""

    name: str
    model: str
    llm: ChatOpenAI
    base_url: str
    api_key: str | None

    @property
    def models_url(self) -> str:
        """The provider's model-listing endpoint."""
        return models_url(self.name, self.base_url)


class FreeLLMRouter:
    """
    Routes LLM calls across multiple free providers with automatic failover.

    When one provider hits rate limits (429), the router tries the next
    provider in the chain. This gives you effectively unlimited free
    inference by pooling across provider tiers.

    Provider chain (by default):
      1. KeyPool (your gateway)
      2. OpenRouter :free models
      3. Groq free tier
      4. Google Gemini free tier
      5. Cloudflare Workers AI free tier
      6. Local Ollama (if running)
    """

    def __init__(self) -> None:
        self.settings = settings
        self._providers: list[Provider] = []
        self._current_index = 0
        self._build_chain()

    def reset(self) -> None:
        """Rebuild the chain after settings changed (e.g. CLI overrides)."""
        self._providers.clear()
        self._current_index = 0
        self._build_chain()

    def _add(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Append one provider to the chain."""
        effort, max_tokens = THINKING_LEVELS.get(
            self.settings.thinking_level, THINKING_LEVELS["medium"]
        )
        # Both go in extra_body so the gateway receives the field names it
        # accepts (max_tokens, reasoning_effort) rather than client-side aliases.
        extra_body: dict[str, object] = {"max_tokens": max_tokens}
        if effort:
            extra_body["reasoning_effort"] = effort

        client = ChatOpenAI(
            model=model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            temperature=0.1,
            default_headers=headers,
            extra_body=extra_body,
            # A hung connection must fail so retry/failover can act on it.
            timeout=self.settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._providers.append(
            Provider(
                name=name,
                model=model,
                llm=client,
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key,
            )
        )

    def _build_chain(self) -> None:
        """Build the provider chain: the configured primary, plus fallbacks if enabled."""
        s = self.settings

        if s.keypool_api_key:
            headers = None
            if s.keypool_protocol.lower() != "openai":
                headers = {"X-Protocol": s.keypool_protocol}
            self._add(
                "keypool",
                s.keypool_model,
                s.keypool_api_key,
                s.keypool_base_url,
                headers,
            )

        if s.openrouter_api_key:
            self._add(
                "openrouter",
                s.openrouter_free_model,
                s.openrouter_api_key,
                s.openrouter_base_url,
            )

        if s.groq_api_key:
            self._add("groq", s.groq_free_model, s.groq_api_key, s.groq_base_url)

        if s.google_api_key:
            self._add("google", s.google_free_model, s.google_api_key, s.google_base_url)

        if s.cloudflare_api_key and s.cloudflare_account_id:
            self._add(
                "cloudflare",
                s.cloudflare_free_model,
                s.cloudflare_api_key,
                f"https://api.cloudflare.com/client/v4/accounts/{s.cloudflare_account_id}/ai/v1",
            )

        if s.openai_api_key:
            self._add("openai", s.openai_model, s.openai_api_key)

        # Anthropic speaks its own protocol; add it through ChatAnthropic if needed.

        # The local server is always the last resort.
        if s.local_llm_api_key:
            self._add("local", s.local_llm_model, s.local_llm_api_key, s.local_llm_base_url)

        if not s.enable_free_fallback:
            # One provider, no drift: every call goes to LLM_PROVIDER.
            self._providers = [p for p in self._providers if p.name == s.llm_provider]

    def get_llm(self, provider_name: str | None = None) -> ChatOpenAI:
        """Get an LLM for a specific provider, or the primary one."""
        if not self._providers:
            raise RuntimeError(
                f"LLM_PROVIDER={settings.llm_provider!r} is not configured. "
                f"Set its API key in .env, or set ENABLE_FREE_FALLBACK=true to use "
                f"any other provider you have keys for."
            )
        if provider_name:
            for p in self._providers:
                if p.name == provider_name:
                    return p.llm
            raise ValueError(f"Provider '{provider_name}' not in chain")

        for i, p in enumerate(self._providers):
            if p.name == settings.llm_provider:
                self._current_index = i
                return p.llm

        self._current_index = 0
        return self._providers[0].llm

    def next_provider(self) -> ChatOpenAI:
        """Advance to the next provider in the chain (for failover)."""
        if not self._providers:
            raise RuntimeError("No providers configured")
        self._current_index = (self._current_index + 1) % len(self._providers)
        return self._providers[self._current_index].llm

    def current_provider_name(self) -> str:
        """Get the name of the current provider."""
        if not self._providers:
            return "none"
        return self._providers[self._current_index].name

    def provider_names(self) -> list[str]:
        """Get all provider names in chain order."""
        return [p.name for p in self._providers]

    @property
    def providers(self) -> list["Provider"]:
        """The configured chain, in order."""
        return list(self._providers)

    async def check(self, provider: "Provider") -> bool:
        """Whether one provider answers its model-listing endpoint."""
        headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(provider.models_url, headers=headers)
            return response.status_code == HTTP_OK
        except Exception:
            return False

    async def check_all(self) -> dict[str, bool]:
        """Check connectivity for all providers."""
        results: dict[str, bool] = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for p in self._providers:
                headers = {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}
                try:
                    resp = await client.get(p.models_url, headers=headers)
                    results[p.name] = resp.status_code == HTTP_OK
                except Exception:
                    results[p.name] = False
        return results


# ── Global router instance ────────────────────────────────────────────
router = FreeLLMRouter()


def check_local_llm() -> bool:
    """Check if a local LLM server is reachable."""
    try:
        response = httpx.get(
            models_url("local", settings.local_llm_base_url),
            timeout=5.0,
        )
        return response.status_code == HTTP_OK
    except Exception:
        return False


def check_keypool() -> bool:
    """Check if KeyPool gateway is reachable."""
    try:
        response = httpx.get(
            models_url("keypool", settings.keypool_base_url),
            headers={"Authorization": f"Bearer {settings.keypool_api_key}"},
            timeout=5.0,
        )
        return response.status_code == HTTP_OK
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_tracer() -> Any:
    """Get the process-wide tracer, setting up observability on first use."""
    # Imported lazily: src.observability reads these settings at import time.
    from src.observability import setup_observability

    return setup_observability()
