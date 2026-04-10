from pathlib import Path


def test_generated_string_fixtures_are_nonempty(
    adversarial_strings: tuple[str, ...],
    adversarial_strings_path: Path,
    fuzz_strings: tuple[str, ...],
    fuzz_strings_path: Path,
) -> None:
    assert adversarial_strings
    assert fuzz_strings
    assert "\n" in adversarial_strings
    assert adversarial_strings_path.exists()
    assert fuzz_strings_path.exists()


def test_gutenberg_fixtures_are_available(
    gutenberg_books: dict[str, str],
    gutenberg_book_paths: dict[str, Path],
) -> None:
    assert sorted(gutenberg_books) == [
        "importance_of_being_earnest",
        "pride_and_prejudice",
        "war_and_peace",
    ]
    assert "War and Peace" in gutenberg_books["war_and_peace"]
    assert "The Importance of Being Earnest" in gutenberg_books["importance_of_being_earnest"]
    assert "Pride and Prejudice" in gutenberg_books["pride_and_prejudice"]
    assert all(path.exists() for path in gutenberg_book_paths.values())


def test_model_fixtures_are_available(
    model_dir: Path,
    odin_id_dir: Path,
) -> None:
    assert (model_dir / "tokenizer.json").exists()
    assert (odin_id_dir / "tokenizer.json").exists()
