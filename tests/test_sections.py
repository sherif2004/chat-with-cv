import pytest

from cv_chat.processing.sections import SECTION_TYPES, section_type


@pytest.mark.parametrize("heading, expected", [
    ("Work Experience", "experience"),
    ("PROFESSIONAL EXPERIENCE", "experience"),
    ("Employment History", "experience"),
    ("Internships", "experience"),
    ("Education", "education"),
    ("University of Cairo", "education"),
    ("Technical Skills", "skills"),
    ("Skillset", "skills"),
    ("Tools & Technologies", "skills"),
    ("Programming Languages", "skills"),
    ("Languages", "languages"),
    ("Spoken Languages", "languages"),
    ("Projects", "projects"),
    ("Personal Projects", "projects"),
    ("Professional Summary", "summary"),
    ("About Me", "summary"),
    ("Certifications & Licenses", "certifications"),
    ("Awards", "certifications"),
    ("Contact", "contact"),
    ("Hobbies", "other"),
    ("Other", "other"),
])
def test_headings_map_to_a_standard_section(heading, expected):
    assert section_type(heading) == expected


@pytest.mark.parametrize("heading", ["Reskilling plan", "Roundabout", "Preschool", "Courses", "Personal Details"])
def test_a_keyword_must_start_a_word_and_vague_keywords_are_gone(heading):
    assert section_type(heading) == "other"


def test_the_section_types_end_with_other():
    assert SECTION_TYPES[-1] == "other" and "experience" in SECTION_TYPES
