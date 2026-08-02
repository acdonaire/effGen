"""
Configuration loader with advanced features.

Features:
- Environment variable substitution (${VAR_NAME} or ${VAR_NAME:default})
- Multiple configuration paths and merging
- YAML and JSON support
- Hot reloading with file watchers
- Secret manager integration (AWS Secrets Manager, HashiCorp Vault)
- Configuration validation
- Thread-safe operations
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import yaml

try:
    import json5 as json
except ImportError:
    import json

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration container with dict-like access.

    Supports both attribute and dictionary-style access:
    - config.models.phi3_mini
    - config["models"]["phi3_mini"]
    """

    data: dict[str, Any] = field(default_factory=dict)
    _source_file: Path | None = None
    _loaded_at: datetime = field(default_factory=datetime.now)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style assignment."""
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self.data.get(key, default)

    def __getattr__(self, name: str) -> Any:
        """Attribute-style access."""
        if name.startswith("_") or name == "data":
            return object.__getattribute__(self, name)

        if name in self.data:
            value = self.data[name]
            # Convert nested dicts to Config objects
            if isinstance(value, dict):
                return Config(data=value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.data.copy()

    def update(self, other: dict | Config) -> None:
        """Update configuration with another config or dict."""
        if isinstance(other, Config):
            self._merge_recursive(self.data, other.data)
        else:
            self._merge_recursive(self.data, other)

    @staticmethod
    def _merge_recursive(base: dict, update: dict) -> None:
        """Recursively merge update into base."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._merge_recursive(base[key], value)
            else:
                base[key] = value


class ConfigFileHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """
    File system event handler for configuration hot reloading.
    """

    def __init__(self, loader: "ConfigLoader", callback: Callable | None = None) -> None:
        """
        Initialize handler.

        Args:
            loader: ConfigLoader instance
            callback: Optional callback function to call on config change
        """
        self.loader = loader
        self.callback = callback
        self.last_modified = {}

    def on_modified(self, event: Any) -> None:
        """Handle file modification event."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if this is a config file we're watching
        if not (file_path.suffix in [".yaml", ".yml", ".json"] and
                file_path in self.loader._watched_files):
            return

        # Debounce - ignore rapid successive changes
        now = datetime.now()
        last = self.last_modified.get(file_path)
        if last and (now - last).total_seconds() < 1:
            return

        self.last_modified[file_path] = now

        logger.info(f"Config file changed: {file_path}")

        try:
            # Reload configuration
            self.loader.reload()

            # Call callback if provided
            if self.callback:
                self.callback(self.loader.config)

        except Exception as e:
            logger.error(f"Failed to reload config after file change: {e}")


class ConfigLoader:
    """
    Advanced configuration loader.

    Features:
    - Environment variable substitution
    - Multiple config file merging
    - YAML and JSON support
    - Hot reloading
    - Secret manager integration
    - Configuration validation

    Example:
        >>> loader = ConfigLoader()
        >>> loader.load_config("configs/models.yaml")
        >>> model = loader.config.models.phi3_mini
    """

    # Regex for environment variable substitution
    # Matches ${VAR_NAME} or ${VAR_NAME:default_value}
    ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

    def __init__(
        self,
        config_dir: str | Path | None = None,
        env_prefix: str = "",
        auto_reload: bool = False,
        reload_callback: Callable | None = None,
        use_secret_manager: bool = False,
        secret_manager_type: str | None = None,
    ) -> None:
        """
        Initialize ConfigLoader.

        Args:
            config_dir: Default directory for config files
            env_prefix: Prefix for environment variables
            auto_reload: Enable automatic reloading on file changes
            reload_callback: Callback function called after reload
            use_secret_manager: Enable secret manager integration
            secret_manager_type: Type of secret manager (aws, vault, azure)
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "configs"
        self.env_prefix = env_prefix
        self.auto_reload = auto_reload
        self.reload_callback = reload_callback
        self.use_secret_manager = use_secret_manager
        self.secret_manager_type = secret_manager_type

        self.config = Config()
        self._config_files: list[Path] = []
        self._watched_files: set = set()
        self._observer: Observer | None = None
        self._lock = threading.RLock()

        # Initialize secret manager if enabled
        self._secret_manager = None
        if use_secret_manager and secret_manager_type:
            self._init_secret_manager(secret_manager_type)

    def _init_secret_manager(self, manager_type: str) -> None:
        """
        Initialize secret manager client.

        Args:
            manager_type: Type of secret manager (aws, vault, azure)
        """
        try:
            if manager_type == "aws":
                import boto3
                self._secret_manager = boto3.client("secretsmanager")
                logger.info("AWS Secrets Manager initialized")
            elif manager_type == "vault":
                import hvac
                vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")
                vault_token = os.getenv("VAULT_TOKEN")
                self._secret_manager = hvac.Client(url=vault_addr, token=vault_token)
                logger.info("HashiCorp Vault initialized")
            elif manager_type == "azure":
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient
                vault_url = os.getenv("AZURE_KEYVAULT_URL")
                if vault_url:
                    credential = DefaultAzureCredential()
                    self._secret_manager = SecretClient(
                        vault_url=vault_url,
                        credential=credential
                    )
                    logger.info("Azure Key Vault initialized")
            else:
                logger.warning(f"Unknown secret manager type: {manager_type}")
        except Exception as e:
            logger.error(f"Failed to initialize secret manager: {e}")
            self._secret_manager = None

    def load_config(
        self,
        config_path: str | Path | list[str | Path],
        merge: bool = True,
        validate: bool = True,
    ) -> Config:
        """
        Load configuration from file(s).

        Args:
            config_path: Path to config file or list of paths
            merge: Merge with existing config if True
            validate: Validate config after loading

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        with self._lock:
            # Handle multiple config files
            if isinstance(config_path, list | tuple):
                configs = []
                for path in config_path:
                    configs.append(self._load_single_config(path))

                # Merge all configs
                merged = Config()
                for cfg in configs:
                    merged.update(cfg)

                if merge:
                    self.config.update(merged)
                else:
                    self.config = merged
                    self._config_files = [self._resolve_path(p) for p in config_path]
            else:
                # Single config file
                loaded = self._load_single_config(config_path)

                if merge:
                    self.config.update(loaded)
                else:
                    self.config = loaded
                    self._config_files = [self._resolve_path(config_path)]

            # Validate if requested
            if validate:
                self.validate_config()

            # Set up hot reloading if enabled
            if self.auto_reload and not self._observer:
                self._setup_hot_reload()

            return self.config

    def load(
        self,
        config_path: str | Path | list[str | Path],
        merge: bool = True,
        validate: bool = True,
    ) -> Config:
        """Alias for :meth:`load_config`.

        Provided because ``load`` is the more obvious method name; it forwards
        directly to :meth:`load_config`.
        """
        return self.load_config(config_path, merge=merge, validate=validate)

    def _load_single_config(self, config_path: str | Path) -> Config:
        """
        Load a single configuration file.

        Args:
            config_path: Path to config file

        Returns:
            Loaded configuration
        """
        path = self._resolve_path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.info(f"Loading config from: {path}")

        # Load file content
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Substitute environment variables
        content = self._substitute_env_vars(content)

        # Parse based on file extension (handle .example suffix)
        # Support both single extensions (.yaml, .json) and double extensions (.yaml.example)
        file_name = path.name.lower()
        if file_name.endswith(('.yaml', '.yml', '.yaml.example', '.yml.example')):
            data = yaml.safe_load(content)
        elif file_name.endswith(('.json', '.json.example')):
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        # A config document is a mapping of settings. Anything else (a bare list,
        # a scalar, a stray document separator) would load here and then fail
        # somewhere far away — on a merge, a lookup, or a validation pass — so it
        # is refused at the file that caused it.
        if data is not None and not isinstance(data, dict):
            raise ValueError(
                f"Config file {path} must contain a mapping of settings at the top "
                f"level, got {type(data).__name__}."
            )

        config = Config(data=data or {}, _source_file=path)

        # Add to watched files
        self._watched_files.add(path)

        return config

    def _resolve_path(self, config_path: str | Path) -> Path:
        """
        Resolve configuration file path.

        Args:
            config_path: Relative or absolute path

        Returns:
            Resolved absolute path
        """
        path = Path(config_path)

        if path.is_absolute():
            return path

        # Try relative to config_dir
        candidate = self.config_dir / path
        if candidate.exists():
            return candidate

        # Try relative to current directory
        candidate = Path.cwd() / path
        if candidate.exists():
            return candidate

        # Return path relative to config_dir (even if doesn't exist)
        return self.config_dir / path

    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.

        Supports:
        - ${VAR_NAME} - substitute with environment variable
        - ${VAR_NAME:default} - use default if variable not set

        Args:
            content: Configuration file content

        Returns:
            Content with substituted variables
        """
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2)

            # Add prefix if configured
            full_var_name = f"{self.env_prefix}{var_name}" if self.env_prefix else var_name

            # Try environment variable first
            value = os.getenv(full_var_name)

            # Try secret manager if enabled and env var not found
            if value is None and self._secret_manager:
                value = self._get_secret(var_name)

            # Use default value if provided
            if value is None and default_value is not None:
                value = default_value

            # Keep original placeholder if no value found
            if value is None:
                logger.warning(f"Environment variable not found: {var_name}")
                return match.group(0)

            return str(value)

        return self.ENV_VAR_PATTERN.sub(replace_var, content)

    def _get_secret(self, secret_name: str) -> str | None:
        """
        Retrieve secret from secret manager.

        Args:
            secret_name: Name of the secret

        Returns:
            Secret value or None if not found
        """
        try:
            if self.secret_manager_type == "aws":
                response = self._secret_manager.get_secret_value(SecretId=secret_name)
                return response.get("SecretString")

            elif self.secret_manager_type == "vault":
                mount_point = os.getenv("VAULT_MOUNT_POINT", "secret")
                secret_path = os.getenv("VAULT_SECRET_PATH", "effgen/")
                full_path = f"{secret_path}{secret_name}"
                response = self._secret_manager.secrets.kv.v2.read_secret_version(
                    path=full_path,
                    mount_point=mount_point
                )
                return response["data"]["data"].get(secret_name)

            elif self.secret_manager_type == "azure":
                secret = self._secret_manager.get_secret(secret_name)
                return secret.value

        except Exception as e:
            logger.debug(f"Failed to retrieve secret '{secret_name}': {e}")

        return None

    def _setup_hot_reload(self) -> None:
        """Set up file watching for hot reload."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not installed. Hot reload disabled.")
            return

        if self._observer:
            return

        # Create event handler
        handler = ConfigFileHandler(self, self.reload_callback)

        # Create observer
        self._observer = Observer()

        # Watch all directories containing config files
        watched_dirs = set()
        for config_file in self._watched_files:
            directory = config_file.parent
            if directory not in watched_dirs:
                self._observer.schedule(handler, str(directory), recursive=False)
                watched_dirs.add(directory)
                logger.info(f"Watching directory for config changes: {directory}")

        # Start observer
        self._observer.start()
        logger.info("Hot reload enabled")

    def reload(self) -> Config:
        """
        Reload configuration from source files.

        Returns:
            Reloaded configuration
        """
        with self._lock:
            logger.info("Reloading configuration...")

            if not self._config_files:
                logger.warning("No config files to reload")
                return self.config

            # Reload all config files
            return self.load_config(self._config_files, merge=False, validate=True)

    def validate_config(self) -> bool:
        """
        Validate loaded configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        from .validator import ConfigValidator

        validator = ConfigValidator()

        # Validate different config sections
        if "models" in self.config:
            validator.validate_models(self.config["models"])

        if "tools" in self.config:
            validator.validate_tools(self.config["tools"])

        if "prompts" in self.config or "system_prompts" in self.config:
            validator.validate_prompts(self.config.to_dict())

        logger.info("Configuration validation passed")
        return True

    def save_config(self, output_path: str | Path, format: str = "yaml") -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Output file path
            format: Output format (yaml or json)
        """
        path = Path(output_path)

        with open(path, "w", encoding="utf-8") as f:
            if format == "yaml":
                yaml.dump(self.config.to_dict(), f, default_flow_style=False)
            elif format == "json":
                json.dump(self.config.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Configuration saved to: {path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Example:
            >>> loader.get("models.phi3_mini.temperature")
            0.7

        Args:
            key_path: Dot-separated path to value
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.config.data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Example:
            >>> loader.set("models.phi3_mini.temperature", 0.8)

        Args:
            key_path: Dot-separated path to value
            value: Value to set
        """
        keys = key_path.split(".")
        target = self.config.data

        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        target[keys[-1]] = value

    def stop(self) -> None:
        """Stop file watching and clean up resources."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Hot reload stopped")

    def __enter__(self) -> ConfigLoader:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit."""
        self.stop()
        return False
