from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pytest

from tokcleanse import (
    DEFAULT_SAVE_ORDER_NAME,
    TokenizerContents,
    load_tokenizer_contents,
    save_reordered_tokenizer,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / "models"
_ORIGINAL_MODEL_ID = "google/gemma-4-E2B-it"
_MODEL_DIR = _PROJECT_ROOT / "models" / "google" / "gemma-4-E2B-it"
_ODIN_ID_DIR = _PROJECT_ROOT / "models" / "odin-id"
_DATA_DIR = _PROJECT_ROOT / "tests" / "data"
_ADVERSARIAL_STRINGS_PATH = _DATA_DIR / "adversarial_strings.json"
_FUZZ_STRINGS_PATH = _DATA_DIR / "fuzz_strings.json"
_GUTENBERG_DIR = _PROJECT_ROOT / "tests" / "data" / "gutenberg"
_ADVERSARIAL_CACHE_VERSION = 1
_FUZZ_CACHE_VERSION = 1
_FUZZ_SEED = 42
_FUZZ_COUNT = 2048
_FUZZ_MAX_PARTS = 8
_FUZZ_MAX_LENGTH = 96
_LOCK_TIMEOUT_SECONDS = 300.0
_LOCK_POLL_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class GutenbergBook:
    key: str
    title: str
    author: str
    url: str
    path: Path


_GUTENBERG_BOOKS: tuple[GutenbergBook, ...] = (
    GutenbergBook(
        key="war_and_peace",
        title="War and Peace",
        author="Leo Tolstoy",
        url="https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
        path=_GUTENBERG_DIR / "war_and_peace.txt",
    ),
    GutenbergBook(
        key="importance_of_being_earnest",
        title="The Importance of Being Earnest",
        author="Oscar Wilde",
        url="https://www.gutenberg.org/cache/epub/844/pg844.txt",
        path=_GUTENBERG_DIR / "the_importance_of_being_earnest.txt",
    ),
    GutenbergBook(
        key="pride_and_prejudice",
        title="Pride and Prejudice",
        author="Jane Austen",
        url="https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        path=_GUTENBERG_DIR / "pride_and_prejudice.txt",
    ),
)


@pytest.fixture(scope="session")
def model_dir() -> Path:
    return load_tokenizer_contents(_ORIGINAL_MODEL_ID, models_dir=_MODELS_DIR).resolved_dir


@pytest.fixture(scope="session")
def sample_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tokenizer_dir = tmp_path_factory.mktemp("sample_tokenizer")
    (tokenizer_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "<pad>": 0,
                        "x": 1,
                        "y": 2,
                        "a": 3,
                        "b": 4,
                        "c": 5,
                        "z": 6,
                        "xy": 7,
                        "ab": 8,
                        "abc": 9,
                        "xyz": 10,
                    },
                    "merges": [
                        ["x", "y"],
                        ["a", "b"],
                        ["ab", "c"],
                        ["xy", "z"],
                    ],
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<pad>"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (tokenizer_dir / "merges.txt").write_text(
        "#version: 0.2\nx y\na b\nab c\nxy z\n",
        encoding="utf-8",
    )
    return tokenizer_dir


@pytest.fixture(scope="session")
def sample_tokenizer_contents(sample_tokenizer_dir: Path) -> TokenizerContents:
    return load_tokenizer_contents(sample_tokenizer_dir)


@pytest.fixture(scope="session")
def tokenizer_contents(model_dir: Path) -> TokenizerContents:
    return load_tokenizer_contents(model_dir)


@pytest.fixture(scope="session")
def odin_id_dir(tokenizer_contents: TokenizerContents) -> Path:
    if not _ODIN_ID_DIR.exists():
        with _file_lock(_ODIN_ID_DIR):
            if not _ODIN_ID_DIR.exists():
                save_reordered_tokenizer(
                    tokenizer_contents,
                    _ODIN_ID_DIR,
                    order_name=DEFAULT_SAVE_ORDER_NAME,
                )
    return _ODIN_ID_DIR


@pytest.fixture(scope="session")
def odin_id_contents(odin_id_dir: Path) -> TokenizerContents:
    return load_tokenizer_contents(odin_id_dir)


@pytest.fixture(scope="session")
def adversarial_strings(tokenizer_contents: TokenizerContents) -> tuple[str, ...]:
    return _load_or_create_adversarial_strings(tokenizer_contents)


@pytest.fixture(scope="session")
def fuzz_strings(tokenizer_contents: TokenizerContents) -> tuple[str, ...]:
    return _load_or_create_fuzz_strings(tokenizer_contents)


@pytest.fixture(scope="session")
def gutenberg_books() -> dict[str, str]:
    _GUTENBERG_DIR.mkdir(parents=True, exist_ok=True)
    books: dict[str, str] = {}
    for book in _GUTENBERG_BOOKS:
        if not book.path.exists():
            with _file_lock(book.path):
                if not book.path.exists():
                    _download_text(url=book.url, destination=book.path)
        books[book.key] = book.path.read_text(encoding="utf-8-sig")
    return books


@pytest.fixture(scope="session")
def gutenberg_book_paths() -> dict[str, Path]:
    return {book.key: book.path for book in _GUTENBERG_BOOKS}


@pytest.fixture(scope="session")
def adversarial_strings_path() -> Path:
    return _ADVERSARIAL_STRINGS_PATH


@pytest.fixture(scope="session")
def fuzz_strings_path() -> Path:
    return _FUZZ_STRINGS_PATH


def _load_or_create_adversarial_strings(contents: TokenizerContents) -> tuple[str, ...]:
    cache = _load_json_file(_ADVERSARIAL_STRINGS_PATH)
    cached_strings = cache.get("strings") if cache is not None else None
    if (
        cache is not None
        and cache.get("cache_version") == _ADVERSARIAL_CACHE_VERSION
        and isinstance(cached_strings, list)
    ):
        strings = tuple(text for text in cached_strings if isinstance(text, str))
        if len(strings) == len(cached_strings):
            return strings

    with _file_lock(_ADVERSARIAL_STRINGS_PATH):
        cache = _load_json_file(_ADVERSARIAL_STRINGS_PATH)
        cached_strings = cache.get("strings") if cache is not None else None
        if (
            cache is not None
            and cache.get("cache_version") == _ADVERSARIAL_CACHE_VERSION
            and isinstance(cached_strings, list)
        ):
            strings = tuple(text for text in cached_strings if isinstance(text, str))
            if len(strings) == len(cached_strings):
                return strings

        strings = _generate_adversarial_strings(contents)
        _write_json_file(
            _ADVERSARIAL_STRINGS_PATH,
            {
                "cache_version": _ADVERSARIAL_CACHE_VERSION,
                "generator": "adversarial_strings",
                "strings": list(strings),
            },
        )
        return strings


def _load_or_create_fuzz_strings(contents: TokenizerContents) -> tuple[str, ...]:
    cache = _load_json_file(_FUZZ_STRINGS_PATH)
    cached_strings = cache.get("strings") if cache is not None else None
    if (
        cache is not None
        and cache.get("cache_version") == _FUZZ_CACHE_VERSION
        and cache.get("seed") == _FUZZ_SEED
        and cache.get("count") == _FUZZ_COUNT
        and cache.get("max_parts") == _FUZZ_MAX_PARTS
        and cache.get("max_length") == _FUZZ_MAX_LENGTH
        and isinstance(cached_strings, list)
    ):
        strings = tuple(text for text in cached_strings if isinstance(text, str))
        if len(strings) == len(cached_strings):
            return strings

    with _file_lock(_FUZZ_STRINGS_PATH):
        cache = _load_json_file(_FUZZ_STRINGS_PATH)
        cached_strings = cache.get("strings") if cache is not None else None
        if (
            cache is not None
            and cache.get("cache_version") == _FUZZ_CACHE_VERSION
            and cache.get("seed") == _FUZZ_SEED
            and cache.get("count") == _FUZZ_COUNT
            and cache.get("max_parts") == _FUZZ_MAX_PARTS
            and cache.get("max_length") == _FUZZ_MAX_LENGTH
            and isinstance(cached_strings, list)
        ):
            strings = tuple(text for text in cached_strings if isinstance(text, str))
            if len(strings) == len(cached_strings):
                return strings

        strings = _generate_fuzz_strings(contents)
        _write_json_file(
            _FUZZ_STRINGS_PATH,
            {
                "cache_version": _FUZZ_CACHE_VERSION,
                "generator": "fuzz_strings",
                "seed": _FUZZ_SEED,
                "count": _FUZZ_COUNT,
                "max_parts": _FUZZ_MAX_PARTS,
                "max_length": _FUZZ_MAX_LENGTH,
                "strings": list(strings),
            },
        )
        return strings


def _generate_adversarial_strings(contents: TokenizerContents) -> tuple[str, ...]:
    cases: list[str] = [
        "",
        " ",
        "\n",
        "\n\n",
        "\t",
        "\t\t",
        "▁",
        "▁▁",
    ]

    seen: set[str] = set(cases)

    def add(text: str) -> None:
        if not text or len(text) > 96 or text in seen:
            return
        seen.add(text)
        cases.append(text)

    ambiguous: dict[str, list[tuple[str, str]]] = {}
    for left, right in contents.original_merges:
        ambiguous.setdefault(left + right, []).append((left, right))

    for merged, parents in ambiguous.items():
        if len(parents) < 2:
            continue
        add(merged)
        for left, right in parents[:3]:
            add(left + right)
            add(left + right + right)
            add(left + merged)
            add(merged + right)
        if len(cases) >= 512:
            break

    for left, right in contents.original_merges[:1024]:
        merged = left + right
        add(merged)
        add(left + right + merged)
        add(merged + right)
        add(left + merged)
        if len(cases) >= 1024:
            break

    for first_left, first_right in contents.original_merges[:2048]:
        middle = first_right
        chain = first_left + first_right
        for second_left, second_right in contents.original_merges[:256]:
            if second_left != middle:
                continue
            add(chain + second_right)
            break
        if len(cases) >= 1536:
            break

    return tuple(cases)


def _generate_fuzz_strings(contents: TokenizerContents) -> tuple[str, ...]:
    pieces = _build_fuzz_pieces(contents)
    rng = random.Random(_FUZZ_SEED)
    strings: list[str] = []
    for _ in range(_FUZZ_COUNT):
        part_count = rng.randint(1, _FUZZ_MAX_PARTS)
        text = "".join(rng.choice(pieces) for _ in range(part_count))
        strings.append(text[:_FUZZ_MAX_LENGTH])
    return tuple(strings)


def _build_fuzz_pieces(contents: TokenizerContents) -> tuple[str, ...]:
    pieces: list[str] = []
    seen: set[str] = set()

    def add(piece: str) -> None:
        if not piece or len(piece) > 12 or piece in seen:
            return
        seen.add(piece)
        pieces.append(piece)

    for token in contents.token_to_index:
        if len(token) == 1:
            add(token)

    for left, right in contents.original_merges[:4096]:
        add(left)
        add(right)
        add(left + right)
        if len(pieces) >= 2048:
            break

    return tuple(pieces)


def _load_json_file(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def _write_json_file(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _download_text(*, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        temp_path.write_bytes(response.read())
        temp_path.replace(destination)


@dataclass
class _FileLock:
    path: Path
    _fd: int | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                self._fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                return None
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for lock {self.path}")
                time.sleep(_LOCK_POLL_INTERVAL_SECONDS)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _file_lock(path: Path) -> _FileLock:
    return _FileLock(path.with_suffix(path.suffix + ".lock"))
