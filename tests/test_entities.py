from cv_chat.processing import entities
from cv_chat.processing.chunking import Chunk

LABELS = entities.LABELS


class FakeModel:
    """Finds each needle (when the asked label matches) and records every question it was asked."""

    def __init__(self, *rules):
        self.rules, self.asked = rules, []  # rules: (label key, needle, score)

    def predict_entities(self, text, labels, threshold=0.5):
        self.asked.append((labels[0], text))
        found = []
        for key, needle, score in self.rules:
            labels_for_key = [LABELS[key], *entities.settings.entities.location_fallback_labels] if key == "location" else [LABELS[key]]
            if labels[0] in labels_for_key and needle in text and score >= threshold:
                start = text.index(needle)
                found.append({"text": needle, "start": start, "end": start + len(needle), "score": score})
        return found

    def asked_labels(self):
        return [label for label, _ in self.asked]


def blocks(*pairs):
    return [Chunk(section, 1, text) for section, text in pairs]


def test_everything_in_the_header_is_read_from_the_first_block_only():
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("job_title", "AI Engineer", 0.8), ("email", "jane@x.com", 0.9),
                      ("phone", "+20 100 123 4567", 0.8), ("location", "Cairo", 0.8))
    found = entities.extract(blocks(("Other", "Jane Doe\nAI Engineer\njane@x.com | +20 100 123 4567 | Cairo"), ("EXPERIENCE", "EXPERIENCE\nBuilt APIs")), model)
    assert found == {"candidate_name": "Jane Doe", "job_title": "AI Engineer", "email": "jane@x.com", "phone": "+20 100 123 4567", "location": "Cairo"}
    assert all("Built APIs" not in text for _, text in model.asked)  # all five were filled, so the next section was never read


def test_a_missing_detail_is_looked_for_in_the_next_sections_in_order_and_the_first_answer_wins():
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("phone", "0100 111", 0.8), ("phone", "0100 222", 0.8))
    found = entities.extract(blocks(("Other", "Jane Doe"), ("CONTACT", "CONTACT\n0100 111"), ("REFERENCES", "REFERENCES\n0100 222")), model)
    assert found["phone"] == "0100 111"


def test_a_detail_that_is_found_is_not_asked_for_again():
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("job_title", "AI Engineer", 0.9))
    entities.extract(blocks(("Other", "Jane Doe\nAI Engineer"), ("SKILLS", "SKILLS\nPython")), model)
    assert model.asked_labels().count(LABELS["candidate_name"]) == 1
    assert model.asked_labels().count(LABELS["job_title"]) == 1


def test_a_job_title_must_come_after_the_name_in_the_same_text():
    text = "Senior Developer\nJane Doe\nML Engineer"
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("job_title", "Senior Developer", 0.9), ("job_title", "ML Engineer", 0.8))
    found = entities.extract(blocks(("Other", text)), model)
    assert found["job_title"] == "ML Engineer"


def test_a_short_line_shared_by_name_and_title_gives_the_title_without_the_name():
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("job_title", "AI Engineer", 0.8))
    found = entities.extract(blocks(("Other", "Jane Doe | Junior AI Engineer")), model)
    assert found["job_title"] == "Junior AI Engineer" and found["candidate_name"] == "Jane Doe"


def test_the_location_is_taken_from_the_header_or_a_summary_but_not_from_education():
    education = ("EDUCATION", "EDUCATION\nCairo University, Giza")
    summary = ("SUMMARY", "SUMMARY\nBased in Alexandria")
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("location", "Giza", 0.9), ("location", "Alexandria", 0.9))
    assert entities.extract(blocks(("Other", "Jane Doe"), education), model)["location"] == ""
    assert entities.extract(blocks(("Other", "Jane Doe"), education, summary), model)["location"] == "Alexandria"


def test_a_cut_word_is_widened_to_the_whole_word_and_markup_is_removed():
    model = FakeModel(("email", "jane@exa", 0.9), ("candidate_name", "Jane <b>Doe</b>", 0.9))
    found = entities.extract(blocks(("Other", "Jane <b>Doe</b>\njane@example.com")), model)
    assert found["email"] == "jane@example.com"
    assert "<" not in found["candidate_name"] and ">" not in found["candidate_name"]


def test_nothing_found_gives_empty_strings():
    assert entities.extract(blocks(("Other", "nothing here")), FakeModel()) == {key: "" for key in LABELS}


def test_a_cv_without_sections_gives_empty_strings():
    assert entities.extract([], FakeModel()) == {key: "" for key in LABELS}
