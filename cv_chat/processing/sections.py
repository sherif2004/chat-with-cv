"""Map the heading a CV uses ("Work History", "Employment", ...) to one of a few standard section types."""
import re

_PATTERNS = {
    "experience": r"experience|employment|work history|career|professional background|internship",
    "education": r"education|academic|qualification|degree|universit|school",
    "skills": r"skill|technolog|competenc|tools|expertise|tech stack",
    "projects": r"project",
    "summary": r"summary|profile|objective|about",
    "certifications": r"certif|licen[sc]e|award|course|training",
    "languages": r"language",
    "contact": r"contact|personal",
}
SECTION_TYPES = [*_PATTERNS, "other"]


def section_type(heading: str) -> str:
    heading = heading.lower()
    return next((name for name, pattern in _PATTERNS.items() if re.search(pattern, heading)), "other")
