"""
Sanity tests for the BM25 build artifacts.

These are invariant checks (shape/consistency) so we can iterate on search quickly
without silently breaking the index format.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# Local import: scripts/ is executed as a directory, so this should resolve
# when running tests from repo root (e.g., `python -m unittest` or `pytest`).
from scripts.common.text_processing import tokenize  


RE_CHUNK_ID_PAGES = re.compile(r"_p(?P<start>\d{4})-p(?P<end>\d{4})_c\d{3}$")

# Keep this in sync with build_bm25.py
MIN_META_KEYS = {
    "iptp_id",
    "title",
    "administration",
    "original_date_announced",
    "effective_date",
    "agencies_affected",
    "subject_matter",
}


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Invalid JSONL at {path}:{line_num}") from e


def _load_terms_and_offsets(inv_path: Path) -> Tuple[List[str], Dict[str, int]]:
    terms: List[str] = []
    for obj in _read_jsonl(inv_path):
        term = obj.get("term")
        if not isinstance(term, str) or not term:
            raise AssertionError(f"Invalid term record in {inv_path}: {obj}")
        terms.append(term)

    return terms, {}


def _is_json_safe_primitive(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool))


class TestBuildBM25Artifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Default artifact locations (matches build_bm25.py defaults).
        cls.out_dir = Path("data/indexes/bm25")
        cls.docs_path = cls.out_dir / "docs.jsonl"
        cls.inv_path = cls.out_dir / "inverted_index.jsonl"
        cls.term_offsets_path = cls.out_dir / "term_offsets.json"
        cls.doc_lens_path = cls.out_dir / "doc_lens.json"
        cls.corpus_stats_path = cls.out_dir / "corpus_stats.json"

        # Ensure artifacts exist.
        for p in [
            cls.docs_path,
            cls.inv_path,
            cls.term_offsets_path,
            cls.doc_lens_path,
            cls.corpus_stats_path,
        ]:
            if not p.exists():
                raise AssertionError(f"Missing BM25 artifact: {p}")

        cls.doc_lens: List[int] = json.loads(cls.doc_lens_path.read_text(encoding="utf-8"))
        cls.corpus_stats: Dict[str, Any] = json.loads(cls.corpus_stats_path.read_text(encoding="utf-8"))
        cls.use_stopwords: bool = bool(cls.corpus_stats.get("use_stopwords", False))

        cls.docs: List[Dict[str, Any]] = list(_read_jsonl(cls.docs_path))
        cls.n_docs = len(cls.docs)

        # Quick shape sanity up front.
        if cls.n_docs == 0:
            raise AssertionError("docs.jsonl is empty")
        if len(cls.doc_lens) != cls.n_docs:
            raise AssertionError(
                f"doc_lens length {len(cls.doc_lens)} != docs count {cls.n_docs}"
            )

    def test_01_doc_len_matches_tokenization_spot_check(self) -> None:
        """
        Spot-check that doc_lens[doc_id] equals the number of tokens produced
        by text_processing.tokenize(...) under the build's stopword flag.
        """
        sample_n = min(self.n_docs, 20)
        for i in range(sample_n):
            d = self.docs[i]
            self.assertIn("doc_id", d)
            self.assertIn("text", d)
            doc_id = int(d["doc_id"])
            text = str(d["text"])
            toks = tokenize(text, use_stopwords=self.use_stopwords)
            self.assertEqual(
                len(toks),
                int(self.doc_lens[doc_id]),
                msg=f"doc_len mismatch for doc_id={doc_id}",
            )

        print(f"PASS: doc_len matches tokenization (spot-check n={sample_n})")

    def test_02_chunk_id_page_bounds_consistency(self) -> None:
        """
        Ensure chunk_id encodes page_start/page_end consistently (p####-p####).
        """
        for d in self.docs:
            chunk_id = str(d.get("chunk_id", ""))
            page_start = int(d.get("page_start"))
            page_end = int(d.get("page_end"))

            m = RE_CHUNK_ID_PAGES.search(chunk_id)
            self.assertIsNotNone(m, msg=f"chunk_id does not match expected format: {chunk_id}")

            start = int(m.group("start"))
            end = int(m.group("end"))
            self.assertEqual(start, page_start, msg=f"page_start mismatch for {chunk_id}")
            self.assertEqual(end, page_end, msg=f"page_end mismatch for {chunk_id}")
            self.assertLessEqual(page_start, page_end, msg=f"inverted page bounds for {chunk_id}")

        print("PASS: chunk_id page bounds match page_start/page_end")

    def test_03_postings_have_unique_doc_ids_spot_check(self) -> None:
        """
        For a sample of terms, ensure postings doc_ids are unique and non-negative.
        (Also catches accidental duplicates within a postings list.)
        """
        sample_terms = 0
        max_terms_to_check = 50

        for obj in _read_jsonl(self.inv_path):
            term = obj.get("term")
            postings = obj.get("postings")

            self.assertIsInstance(term, str)
            self.assertTrue(term)

            self.assertIsInstance(postings, list)
            seen: set[int] = set()

            for pair in postings:
                self.assertIsInstance(pair, list, msg=f"posting must be [doc_id, tf] for term={term}")
                self.assertEqual(len(pair), 2, msg=f"posting must be [doc_id, tf] for term={term}")
                doc_id = int(pair[0])
                tf = int(pair[1])

                self.assertGreaterEqual(doc_id, 0, msg=f"negative doc_id in postings for term={term}")
                self.assertGreater(tf, 0, msg=f"non-positive tf in postings for term={term}")

                self.assertNotIn(doc_id, seen, msg=f"duplicate doc_id in postings for term={term}")
                seen.add(doc_id)

            sample_terms += 1
            if sample_terms >= max_terms_to_check:
                break

        self.assertGreater(sample_terms, 0, msg="inverted_index.jsonl appears empty")
        print(f"PASS: postings doc_ids unique (spot-check terms={sample_terms})")

    def test_04_term_offsets_cover_all_terms_and_seek_correctly(self) -> None:
        """
        term_offsets.json should:
          - contain exactly one offset per term in inverted_index.jsonl
          - allow seeking to the correct term line
        """
        term_offsets: Dict[str, int] = json.loads(self.term_offsets_path.read_text(encoding="utf-8"))
        self.assertIsInstance(term_offsets, dict)
        self.assertGreater(len(term_offsets), 0, msg="term_offsets.json is empty")

        # Collect all terms from inverted_index.jsonl.
        terms: List[str] = []
        for obj in _read_jsonl(self.inv_path):
            term = obj.get("term")
            self.assertIsInstance(term, str)
            self.assertTrue(term)
            terms.append(term)

        self.assertEqual(
            len(term_offsets),
            len(terms),
            msg="term_offsets count does not match number of terms in inverted_index.jsonl",
        )

        # Ensure every term has an offset.
        for t in terms:
            self.assertIn(t, term_offsets, msg=f"missing offset for term={t!r}")

        # Spot-check seeking: verify the term at the offset matches.
        # (Do a small sample for speed.)
        sample_n = min(25, len(terms))
        with self.inv_path.open("r", encoding="utf-8") as f:
            for t in terms[:sample_n]:
                off = int(term_offsets[t])
                self.assertGreaterEqual(off, 0)
                f.seek(off)
                line = f.readline()
                self.assertTrue(line, msg=f"failed to read line at offset for term={t!r}")
                obj = json.loads(line)
                self.assertEqual(obj.get("term"), t, msg=f"offset points to wrong term for {t!r}")

        print(f"PASS: term_offsets cover all terms + seek works (spot-check n={sample_n})")

    def test_05_metadata_schema_guard(self) -> None:
        """
        Ensure meta is a dict, contains only minimal keys, and values are JSON-safe.
        """
        for d in self.docs:
            meta = d.get("meta")
            self.assertIsInstance(meta, dict, msg=f"meta is not a dict for doc_id={d.get('doc_id')}")

            for k in meta.keys():
                self.assertIn(k, MIN_META_KEYS, msg=f"unexpected meta key {k!r}")

            for k, v in meta.items():
                if v is None:
                    continue
                if _is_json_safe_primitive(v):
                    continue
                if isinstance(v, list):
                    for item in v:
                        self.assertTrue(
                            _is_json_safe_primitive(item),
                            msg=f"meta[{k!r}] contains non-primitive list item: {item!r}",
                        )
                    continue
                if isinstance(v, dict):
                    # Shallow dict check: keys must be strings, values primitive-ish.
                    for dk, dv in v.items():
                        self.assertIsInstance(dk, str)
                        self.assertTrue(
                            _is_json_safe_primitive(dv) or dv is None,
                            msg=f"meta[{k!r}] contains non-primitive dict value: {dv!r}",
                        )
                    continue

                self.fail(f"meta[{k!r}] has unsupported type: {type(v)}")

        print("PASS: meta schema is minimal + JSON-safe")

    def test_06_encoding_and_null_byte_sanity(self) -> None:
        """
        Ensure docs.jsonl is valid UTF-8 and doc texts contain no null bytes.
        """
        # Reading with encoding="utf-8" in setUpClass already validates UTF-8.
        # This test adds a null-byte guard for downstream rendering.
        for d in self.docs:
            text = str(d.get("text", ""))
            self.assertNotIn("\x00", text, msg=f"null byte found in doc_id={d.get('doc_id')}")

        print("PASS: encoding sanity + no null bytes in doc texts")


if __name__ == "__main__":
    unittest.main()
