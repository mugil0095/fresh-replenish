from fresh_replenish.config import RAW_DIR, REPORTS_DIR


def test_paths_do_not_depend_on_working_directory():
    assert RAW_DIR.is_absolute()
    assert REPORTS_DIR.is_absolute()
    assert RAW_DIR.parent.parent == REPORTS_DIR.parent  # both live in the repo root
