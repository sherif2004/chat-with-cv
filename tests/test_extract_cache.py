from cv_chat.processing import extract


def touch(directory, *names):
    for name in names:
        (directory / name).write_text("{}")


def test_cached_extractions_of_another_version_are_deleted(tmp_path):
    touch(tmp_path, "aaa-v1.json", "bbb-v1.json", "ccc.json", "aaa-v2.json", "bbb-v2.json")
    assert extract.sweep_old_cache_versions(tmp_path, 2) == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["aaa-v2.json", "bbb-v2.json"]


def test_a_version_that_only_ends_the_same_way_is_not_confused_with_the_current_one(tmp_path):
    touch(tmp_path, "aaa-v12.json", "aaa-v2.json")
    extract.sweep_old_cache_versions(tmp_path, 2)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["aaa-v2.json"]


def test_files_that_are_not_cached_extractions_are_left_alone(tmp_path):
    touch(tmp_path, "notes.txt", "aaa-v2.json")
    assert extract.sweep_old_cache_versions(tmp_path, 2) == 0
    assert (tmp_path / "notes.txt").exists()


def test_a_missing_directory_is_fine(tmp_path):
    assert extract.sweep_old_cache_versions(tmp_path / "nope", 2) == 0


def test_the_sweep_runs_once_per_process(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(extract, "sweep_old_cache_versions", lambda directory, version: calls.append(directory) or 0)
    monkeypatch.setattr(extract, "_swept", False)
    extract.sweep_once()
    extract.sweep_once()
    assert len(calls) == 1
