"""
ContextPilot Configuration
Central configuration management for all services.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class PineconeConfig:
    """Pinecone vector database configuration."""
    api_key: str
    index_name: str
    environment: str = "us-east-1"  # Default region
    dimension: int = 768  # text-embedding-004 dimension
    metric: str = "cosine"
    
    # Namespaces
    docs_namespace: str = "docs"
    normalized_namespace: str = "normalized"


@dataclass(frozen=True)
class GoogleConfig:
    """Google AI configuration for embeddings and generation."""
    api_key: str
    embedding_model: str = "text-embedding-004"
    generation_model: str = "gemini-2.5-flash"


@dataclass(frozen=True)
class FirecrawlConfig:
    """Firecrawl API configuration."""
    api_key: str
    timeout: int = 60
    max_pages: int = 100
    min_interval_seconds: float = 8.0


@dataclass(frozen=True)
class FirestoreConfig:
    """Firestore database configuration."""
    project_id: str
    database_id: str = "(default)"


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant local vector store configuration."""
    path: Path


@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite database configuration (for local fallback)."""
    path: Path


@dataclass(frozen=True)
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@dataclass(frozen=True)
class AuthConfig:
    """API authentication configuration."""
    mode: str
    api_key: str
    enabled: bool = True


@dataclass(frozen=True)
class MultiTenantConfig:
    """Multi-tenant configuration for hosted deployments."""
    enabled: bool = False
    tenants_collection: str = "tenants"


class Config:
    """
    Main configuration class that loads and validates all settings.
    """
    
    def __init__(self):
        self._validate_required_env_vars()
        
        # Base paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Service configurations
        self.pinecone = PineconeConfig(
            api_key=os.getenv("PINECONE_API_KEY", ""),
            index_name=os.getenv("PINECONE_INDEX_NAME", "contextpilot"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        )
        
        self.google = GoogleConfig(
            api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
        
        self.firecrawl = FirecrawlConfig(
            api_key=os.getenv("FIRECRAWL_API_KEY", ""),
            min_interval_seconds=float(os.getenv("FIRECRAWL_MIN_INTERVAL_SECONDS", "8")),
        )
        
        self.firestore = FirestoreConfig(
            project_id=os.getenv("FIREBASE_PROJECT_ID", ""),
            database_id=os.getenv("FIRESTORE_DATABASE_ID", "(default)"),
        )
        
        # SQLite fallback for local development
        self.database = DatabaseConfig(
            path=self.data_dir / "contextpilot.db",
        )

        # Local vector store fallback (Qdrant embedded)
        self.qdrant = QdrantConfig(
            path=Path(os.getenv("QDRANT_PATH", str(self.data_dir / "qdrant"))),
        )
        
        self.server = ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "").lower() == "true",
        )
        
        self.auth = AuthConfig(
            mode=self._default_auth_mode(),
            api_key=os.getenv("CONTEXTPILOT_API_KEY", ""),
            enabled=os.getenv("AUTH_ENABLED", "true").lower() == "true",
        )

        self.multi_tenant = MultiTenantConfig(
            enabled=os.getenv("MULTI_TENANT_ENABLED", "").lower() == "true",
        )

        self.vector_store_provider = os.getenv("VECTOR_STORE_PROVIDER", "")
        self.default_exclude_paths = self._load_default_exclude_paths()

        # Fail fast in hosted mode if auth is required but not configured.
        if self.is_cloud_run and self.auth.enabled and self.auth.mode == "api_key" and not self.auth.api_key:
            raise EnvironmentError(
                "AUTH is enabled in Cloud Run but CONTEXTPILOT_API_KEY is not set. "
                "Set CONTEXTPILOT_API_KEY (or switch AUTH_MODE=firebase/none)."
            )

    def _load_default_exclude_paths(self) -> list[str]:
        """Load default exclusion paths for sitemap crawling."""
        raw = os.getenv("DEFAULT_EXCLUDE_PATHS", "").strip()
        if raw:
            return [p.strip() for p in raw.split(",") if p.strip()]
        return [
            "/blog",
            "/changelog",
            "/community",
            "/competition",
            "/careers",
            "/pricing",
            "/login",
            "/signup",
            "/auth",
            "/terms",
            "/privacy",
            "/support",
            "/about",
        ]
    
    def _validate_required_env_vars(self) -> None:
        """Validate that required environment variables are set."""
        required = ["GOOGLE_API_KEY"]
        missing = [var for var in required if not os.getenv(var)]
        
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please set them in your .env file."
            )
        
        # Warn about optional but recommended vars
        recommended = ["PINECONE_API_KEY", "FIRECRAWL_API_KEY"]
        missing_recommended = [var for var in recommended if not os.getenv(var)]
        
        if missing_recommended:
            import warnings
            warnings.warn(
                f"Missing recommended environment variables: {', '.join(missing_recommended)}. "
                "Some features may not work."
            )
    
    @property
    def has_pinecone(self) -> bool:
        """Check if Pinecone is configured."""
        return bool(self.pinecone.api_key)
    
    @property
    def has_firecrawl(self) -> bool:
        """Check if Firecrawl is configured."""
        return bool(self.firecrawl.api_key)
    
    @property
    def has_firestore(self) -> bool:
        """Check if Firestore is configured."""
        return bool(self.firestore.project_id)

    @property
    def has_gemini(self) -> bool:
        """Check if Gemini (Google AI) is configured."""
        return bool(self.google.api_key)
    
    @property
    def has_auth(self) -> bool:
        """Check if API authentication is configured."""
        if not self.auth.enabled:
            return False
        if self.auth.mode == "none":
            return False
        if self.auth.mode == "api_key":
            return bool(self.auth.api_key)
        if self.auth.mode == "firebase":
            # Firebase auth uses bearer tokens; relies on default credentials in Cloud Run.
            return True
        if self.auth.mode == "api_key_or_firebase":
            return bool(self.auth.api_key) or True
        return False

    @property
    def is_cloud_run(self) -> bool:
        """Detect Cloud Run runtime."""
        return bool(os.getenv("K_SERVICE"))

    def _default_auth_mode(self) -> str:
        """
        Determine auth mode.

        - Local dev: default to none (no auth) unless AUTH_MODE provided.
        - Cloud Run: default to api_key unless AUTH_MODE provided.
        """
        explicit = os.getenv("AUTH_MODE", "").strip().lower()
        if explicit:
            return explicit
        return "api_key" if self.is_cloud_run else "none"


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
