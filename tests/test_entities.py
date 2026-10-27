from cv_chat.processing import entities
from cv_chat.processing.chunking import Chunk

LABELS = entities.LABELS


class FakeModel:
    """Finds each needle when the asked label matches, and records every question it was asked."""

    def __init__(self, *rules):
        self.rules, self.asked = rules, []  # rules: (detail, needle, score)

    def predict_entities(self, text, labels, threshold=0.5):
        self.asked.append((labels[0], text))
        found = []
        for key, needle, score in self.rules:
            if labels[0] == LABELS[key] and needle in text and score >= threshold:
                start = text.index(needle)
                found.append({"text": needle, "start": start, "end": start + len(needle), "score": score})
        return found

    def asked_labels(self):
        return [label for label, _ in self.asked]


def blocks(*pairs):
    return [Chunk(section, 1, text) for section, text in pairs]


def test_each_detail_is_the_first_answer_found_reading_the_cv_in_order():
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("phone", "0100 111", 0.8), ("phone", "0100 222", 0.9))
    found = entities.extract(blocks(("Other", "Jane Doe"), ("CONTACT", "CONTACT\n0100 111"), ("REFERENCES", "REFERENCES\n0100 222")), model)
    assert found["candidate_name"] == "Jane Doe" and found["phone"] == "0100 111"


def test_a_detail_that_is_found_is_never_replaced_by_a_later_one():
    model = FakeModel(("job_title", "AI Engineer", 0.6), ("job_title", "Data Scientist", 0.99))
    found = entities.extract(blocks(("Other", "Jane Doe\nAI Engineer"), ("EXPERIENCE", "EXPERIENCE\nData Scientist at Acme")), model)
    assert found["job_title"] == "AI Engineer"


def test_a_detail_that_is_found_is_not_asked_for_again():
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("job_title", "AI Engineer", 0.9))
    entities.extract(blocks(("Other", "Jane Doe\nAI Engineer"), ("SKILLS", "SKILLS\nPython")), model)
    assert model.asked_labels().count(LABELS["candidate_name"]) == 1
    assert model.asked_labels().count(LABELS["job_title"]) == 1


def test_inside_the_first_text_that_has_an_answer_the_most_confident_one_is_taken():
    model = FakeModel(("email", "0100123456", 0.4), ("email", "jane@x.com", 0.9))  # the email label also fires on phone numbers
    assert entities.extract(blocks(("Other", "0100123456 jane@x.com")), model)["email"] == "jane@x.com"


def test_reading_stops_as_soon_as_every_detail_is_filled():
    model = FakeModel(*[(key, f"<{key}>", 0.9) for key in LABELS])
    entities.extract(blocks(("Other", " ".join(f"<{key}>" for key in LABELS)), ("EXPERIENCE", "EXPERIENCE\nnever read")), model)
    assert all("never read" not in text for _, text in model.asked)


def test_a_detail_that_is_missing_keeps_the_reading_going_to_the_end_of_the_cv():
    chunks = blocks(("Other", "Jane Doe"), *[(f"S{n}", f"S{n}\nfiller {n}") for n in range(40)], ("CONTACT", "CONTACT\n0100 999"))
    model = FakeModel(("candidate_name", "Jane Doe", 0.9), ("phone", "0100 999", 0.8))
    assert entities.extract(chunks, model)["phone"] == "0100 999"
    assert any("filler 39" in text for _, text in model.asked)  # no fixed number of windows


def test_a_long_section_is_read_window_by_window():
    long_text = "\n".join(f"line {n} " + "word " * 20 for n in range(40)) + "\n0100 555"
    model = FakeModel(("phone", "0100 555", 0.8))
    assert entities.extract(blocks(("Other", long_text)), model)["phone"] == "0100 555"


def test_values_are_one_clean_line_without_markup():
    model = FakeModel(("candidate_name", "Jane <b>Doe</b>\nSmith", 0.9))
    value = entities.extract(blocks(("Other", "Jane <b>Doe</b>\nSmith")), model)["candidate_name"]
    assert "<" not in value and ">" not in value and "\n" not in value


def test_nothing_found_gives_empty_strings():
    assert entities.extract(blocks(("Other", "nothing here")), FakeModel()) == {key: "" for key in LABELS}


def test_a_cv_without_sections_gives_empty_strings():
    assert entities.extract([], FakeModel()) == {key: "" for key in LABELS}
