"""Where the .env file is looked for. In its own module so config.py and account_config.py agree without importing each other."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILES = (".env", ROOT / ".env")  # the current directory first, then the project folder
