"""The tunables live in the settings. This fails when one of the old module-level constants comes back."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "cv_chat"
FORBIDDEN = {
    "CHUNK_SIZE", "CHUNK_OVERLAP", "CONTEXT_CHARS", "HISTORY_CHARS", "RETRIEVE_K", "TOP_P", "TOP_K", "COVER_ROUNDS", "EXPANDED_QUERIES",
    "RRF_K", "EXTRACT_CACHE_DIR", "CACHE_SIZE", "EMBED_CACHE_SIZE", "AGENT_MAX_ROUNDS", "AGENT_MAX_SECONDS", "NER_MODEL",
    "INDEX_VERSION", "MAX_NAME_LENGTH", "BOOSTED_SECTIONS", "SEMANTIC_CONFIG", "SCORING_PROFILE", "ANALYZER", "THRESHOLD",
    "WINDOW_WORDS", "PAGE_CHARS", "MAX_USERS", "MIN_PASSWORD", "SESSION_DAYS", "DATABASE_URL",
}


def module_level_names(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        names |= {target.id for target in targets if isinstance(target, ast.Name)}
    return names


def test_no_module_defines_a_tunable_of_its_own():
    found = {
        f"{path.relative_to(PACKAGE.parent)}: {name}"
        for path in PACKAGE.rglob("*.py")
        for name in module_level_names(path) & FORBIDDEN
    }
    assert not found, f"move these into cv_chat/config.py or cv_chat/account_config.py: {sorted(found)}"
