# tests/unit/test_citation_verifier.py
from app.models.schemas import Citation, CitationPosition, RetrievedNode
from app.services.citation_verifier import CitationVerifier


def _make_ctx(
    idx: str = "1",
    content: str = "糖尿病诊断标准为空腹血糖>=7.0mmol/L",
    score: float = 0.9,
    source_file: str = "糖尿病诊疗指南.md",
    page_number: int | None = 5,
    doc_id: str | None = "doc-1",
) -> RetrievedNode:
    return RetrievedNode(
        node_id=idx,
        content=content,
        score=score,
        metadata={
            "source_file": source_file,
            "page_number": page_number,
            "doc_id": doc_id,
        },
    )


class TestExtractAndVerifyHappyPath:
    """Tests for extract_and_verify() — happy path and direct citation matching."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_english_bracket_format_matching_citation(self):
        answer = "根据[来源1](糖尿病诊疗指南#5)的标准，空腹血糖>=7.0可诊断为糖尿病。"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md", page_number=5)]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].source_id == "1"
        assert result[0].verified is True
        assert result[0].position == CitationPosition.DIRECT
        assert result[0].page_number == 5

    def test_chinese_quote_format_matching_citation(self):
        answer = "根据「来源1」（糖尿病诊疗指南#5）的标准。"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md", page_number=5)]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].source_id == "1"
        assert result[0].verified is True
        assert result[0].position == CitationPosition.DIRECT
        assert "糖尿病诊疗指南" in result[0].file_name or "糖尿病诊疗指南" in str(result[0].quote_in_answer)


class TestExtractAndVerifyPageParsing:
    """Tests for page number parsing from source descriptions."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_page_number_parsed_and_matched(self):
        answer = "参考[来源1](糖尿病诊疗指南#5)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md", page_number=5)]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].page_number == 5
        assert result[0].verified is True

    def test_page_number_mismatch_unverified(self):
        answer = "参考[来源1](糖尿病诊疗指南#5)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md", page_number=10)]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].verified is False
        assert result[0].page_number == 5

    def test_source_desc_without_page_number(self):
        answer = "参考[来源1](糖尿病诊疗指南)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md", page_number=None)]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].page_number is None
        assert result[0].verified is True

    def test_source_desc_with_hash_but_non_numeric_page(self):
        answer = "参考[来源1](糖尿病诊疗指南#abc)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md", page_number=None)]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].page_number is None  # non-numeric page → None
        assert result[0].verified is True


class TestExtractAndVerifyFuzzyFilename:
    """Tests for fuzzy filename matching."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_exact_filename_match(self):
        answer = "参考[来源1](糖尿病诊疗指南)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md")]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert result[0].verified is True

    def test_substring_filename_match(self):
        answer = "参考[来源1](诊疗指南)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md")]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert result[0].verified is True

    def test_filename_match_with_medical_suffix_removed(self):
        """Filenames with common medical suffixes should still match."""
        answer = "参考[来源1](糖尿病诊疗规范)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md")]

        result = self.verifier.extract_and_verify(answer, contexts)

        # After removing suffixes "规范" and "指南", both become "糖尿病诊疗"
        assert result[0].verified is True

    def test_filename_no_match_different_files(self):
        answer = "参考[来源1](高血压诊疗指南)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病诊疗指南.md")]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert result[0].verified is False
        assert "不匹配" in result[0].verification_message


class TestExtractAndVerifySourceIndexOutOfRange:
    """Tests for hallucination detection when source_index > len(contexts)."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_source_index_out_of_range(self):
        answer = "根据[来源5](某文件#1)所述..."
        contexts = [_make_ctx(idx="1")]  # only 1 context, index 5 is out of range

        result = self.verifier.extract_and_verify(answer, contexts)

        hallucination = [c for c in result if c.source_id == "5"]
        assert len(hallucination) == 1
        assert hallucination[0].verified is False
        assert hallucination[0].position == CitationPosition.UNVERIFIED
        assert "幻觉" in hallucination[0].verification_message
        assert hallucination[0].document_id is None
        assert hallucination[0].chunk_content == ""
        assert hallucination[0].relevance_score == 0.0

    def test_all_sources_out_of_range_with_contexts_present(self):
        answer = "根据[来源10](文件A)和[来源20](文件B)的说法..."
        contexts = [_make_ctx(idx="1"), _make_ctx(idx="2")]

        result = self.verifier.extract_and_verify(answer, contexts)

        _direct_citations = [c for c in result if c.position == CitationPosition.DIRECT]
        hallucination_citations = [c for c in result if c.position == CitationPosition.UNVERIFIED]
        # The two out-of-range source indices are hallucinations
        assert len(hallucination_citations) == 2
        # And the two real contexts appear as INDIRECT (uncited)
        indirect = [c for c in result if c.position == CitationPosition.INDIRECT]
        assert len(indirect) == 2


class TestExtractAndVerifyEmptyInputs:
    """Tests for empty contexts and empty answer."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_empty_contexts_returns_empty_list(self):
        result = self.verifier.extract_and_verify("根据[来源1](文件#5)的内容", [])
        assert result == []

    def test_empty_answer_returns_uncited_contexts(self):
        contexts = [
            _make_ctx(idx="1", source_file="指南A.md"),
            _make_ctx(idx="2", source_file="指南B.md"),
        ]
        result = self.verifier.extract_and_verify("", contexts)

        assert len(result) == 2
        assert all(c.position == CitationPosition.INDIRECT for c in result)
        assert all(c.verified is True for c in result)
        assert all(c.quote_in_answer is None for c in result)
        assert all("未在回答中引用" in c.verification_message for c in result)

    def test_no_citation_patterns_in_answer_marks_all_as_indirect(self):
        """When answer has no citation patterns, all contexts become INDIRECT."""
        answer = "糖尿病是一种代谢性疾病，需要定期监测血糖。"
        contexts = [_make_ctx(idx="1"), _make_ctx(idx="2")]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 2
        assert all(c.position == CitationPosition.INDIRECT for c in result)
        assert all("未在回答中引用" in c.verification_message for c in result)


class TestExtractAndVerifyDuplicateCitations:
    """Tests for duplicate citation handling."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_duplicate_citations_deduplicated(self):
        """When both old and new patterns match the same text, keep only one."""
        # This string intentionally contains a pattern that might match both regexes
        answer = "参考[来源1](糖尿病指南#5)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病指南.md", page_number=5)]

        result = self.verifier.extract_and_verify(answer, contexts)

        # Should only produce one citation for source_id="1" in DIRECT position
        direct = [c for c in result if c.source_id == "1" and c.position == CitationPosition.DIRECT]
        assert len(direct) == 1

    def test_same_source_mentioned_multiple_times(self):
        # Second citation has different page number → won't match context → UNVERIFIED
        answer = "参考[来源1](糖尿病指南#5)以及[来源1](糖尿病指南#6)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病指南.md", page_number=5)]

        result = self.verifier.extract_and_verify(answer, contexts)

        # Two distinct citations (different quote_in_answer) → both kept, not dedup'd
        all_for_source_1 = [c for c in result if c.source_id == "1"]
        # One is DIRECT (page matches), one is UNVERIFIED (page mismatch)
        direct = [c for c in all_for_source_1 if c.position == CitationPosition.DIRECT]
        unverified = [c for c in all_for_source_1 if c.position == CitationPosition.UNVERIFIED]
        assert len(direct) == 1
        assert len(unverified) == 1


class TestExtractAndVerifyChineseQuoteFormat:
    """Tests specifically for Chinese quote bracket format."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_chinese_format_with_page_number(self):
        answer = "根据「来源2」（高血压防治指南#12）的建议。"
        contexts = [
            _make_ctx(idx="1", source_file="dummy.md"),
            _make_ctx(idx="2", source_file="高血压防治指南.md", page_number=12),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) >= 1
        direct = [c for c in result if c.source_id == "2" and c.position == CitationPosition.DIRECT]
        assert len(direct) == 1
        assert direct[0].verified is True
        assert direct[0].page_number == 12

    def test_chinese_format_without_page_number(self):
        answer = "「来源1」（药物手册）推荐使用。"
        contexts = [_make_ctx(idx="1", source_file="药物手册.md", page_number=None)]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.source_id == "1" and c.position == CitationPosition.DIRECT]
        assert len(direct) == 1
        assert direct[0].verified is True

    def test_chinese_format_multiple_citations(self):
        answer = "「来源1」（指南A#3）指出...「来源2」（指南B#5）补充..."
        contexts = [
            _make_ctx(idx="1", source_file="指南A.md", page_number=3),
            _make_ctx(idx="2", source_file="指南B.md", page_number=5),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct_verified = [c for c in result if c.position == CitationPosition.DIRECT and c.verified]
        assert len(direct_verified) == 2


class TestExtractAndVerifySimpleFormat:
    """Tests for simple 「来源X」 format (no parenthesized file name).

    This is the format the prompt actually instructs the LLM to use:
    "使用「来源X」格式". Previously, the verifier only checked patterns
    with parenthesized file info, so all simple-format citations were
    missed and ALL contexts were incorrectly flagged as "未在回答中引用".
    """

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_simple_format_matches_context(self):
        answer = "糖尿病诊断标准为空腹血糖≥7.0mmol/L「来源1」。"
        contexts = [_make_ctx(idx="1", source_file="糖尿病指南.md")]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].source_id == "1"
        assert result[0].verified is True
        assert result[0].position == CitationPosition.DIRECT
        assert result[0].quote_in_answer == "「来源1」"
        assert result[0].verification_message is None

    def test_simple_format_multiple_citations(self):
        answer = "空腹血糖≥7.0「来源1」，餐后2h≥11.1「来源2」。"
        contexts = [
            _make_ctx(idx="1", source_file="指南A.md"),
            _make_ctx(idx="2", source_file="指南B.md"),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.position == CitationPosition.DIRECT]
        assert len(direct) == 2
        # No INDIRECT → all contexts were properly recognized as cited
        indirect = [c for c in result if c.position == CitationPosition.INDIRECT]
        assert len(indirect) == 0

    def test_simple_format_no_false_uncited_message(self):
        """When all sources are cited via simple format, none should show '未引用'."""
        answer = "「来源1」和「来源2」都提到了这一点。"
        contexts = [
            _make_ctx(idx="1", source_file="文件A.md"),
            _make_ctx(idx="2", source_file="文件B.md"),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        uncited = [c for c in result if c.verification_message and "未在回答中引用" in c.verification_message]
        assert len(uncited) == 0

    def test_simple_format_partial_citation(self):
        """Only source 1 cited, source 2 should still show as INDIRECT."""
        answer = "内容来自「来源1」。"
        contexts = [
            _make_ctx(idx="1", source_file="文件A.md"),
            _make_ctx(idx="2", source_file="文件B.md"),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.position == CitationPosition.DIRECT]
        indirect = [c for c in result if c.position == CitationPosition.INDIRECT]
        assert len(direct) == 1
        assert direct[0].source_id == "1"
        assert len(indirect) == 1
        assert indirect[0].source_id == "2"
        assert "未在回答中引用" in indirect[0].verification_message

    def test_simple_format_index_out_of_range(self):
        answer = "根据「来源99」的说法。"
        contexts = [_make_ctx(idx="1")]

        result = self.verifier.extract_and_verify(answer, contexts)

        hallucination = [c for c in result if c.source_id == "99"]
        assert len(hallucination) == 1
        assert hallucination[0].verified is False
        assert hallucination[0].position == CitationPosition.UNVERIFIED
        assert "幻觉" in hallucination[0].verification_message

    def test_simple_format_does_not_duplicate_full_pattern(self):
        """When full pattern 「来源1」（file）matches, simple pattern should not add a duplicate."""
        answer = "根据「来源1」（糖尿病指南#5）的标准。"
        contexts = [_make_ctx(idx="1", source_file="糖尿病指南.md", page_number=5)]

        result = self.verifier.extract_and_verify(answer, contexts)

        # Should only be 1 citation for source 1, not 2
        source1 = [c for c in result if c.source_id == "1"]
        assert len(source1) == 1
        assert source1[0].verified is True


class TestExtractAndVerifyMixedFormats:
    """Tests for mixed citation formats in the same answer."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_mixed_english_and_chinese_formats(self):
        answer = "根据[来源1](指南A#3)和「来源2」（指南B#5）的研究。"
        contexts = [
            _make_ctx(idx="1", source_file="指南A.md", page_number=3),
            _make_ctx(idx="2", source_file="指南B.md", page_number=5),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.position == CitationPosition.DIRECT]
        assert len(direct) == 2

    def test_mixed_with_some_hallucinated(self):
        answer = "[来源1](指南A#3) 和 [来源99](不存在文件) 都提及..."
        contexts = [_make_ctx(idx="1", source_file="指南A.md", page_number=3)]

        result = self.verifier.extract_and_verify(answer, contexts)

        hallucination = [c for c in result if c.source_id == "99"]
        assert len(hallucination) == 1
        assert hallucination[0].verified is False
        assert "幻觉" in hallucination[0].verification_message


class TestExtractAndVerifyUncitedContexts:
    """Tests for uncited context detection (Indirect citations)."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_uncited_context_added_as_indirect(self):
        answer = "根据[来源1](指南A#3)的标准。"
        contexts = [
            _make_ctx(idx="1", source_file="指南A.md", page_number=3),
            _make_ctx(idx="2", source_file="指南B.md", page_number=7),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.position == CitationPosition.DIRECT]
        indirect = [c for c in result if c.position == CitationPosition.INDIRECT]
        assert len(direct) == 1
        assert len(indirect) == 1
        assert indirect[0].source_id == "2"
        assert indirect[0].verified is True
        assert "未在回答中引用" in indirect[0].verification_message
        assert indirect[0].quote_in_answer is None

    def test_all_contexts_cited_no_indirect(self):
        answer = "参考[来源1](指南A#3)以及[来源2](指南B#5)。"
        contexts = [
            _make_ctx(idx="1", source_file="指南A.md", page_number=3),
            _make_ctx(idx="2", source_file="指南B.md", page_number=5),
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        indirect = [c for c in result if c.position == CitationPosition.INDIRECT]
        assert len(indirect) == 0


class TestExtractCitationsOnly:
    """Tests for extract_citations_only() method."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_extracts_citations_from_context_dicts(self):
        contexts = [
            {
                "node_id": "n1",
                "source": "糖尿病指南.md",
                "page": 5,
                "content": "糖尿病诊断标准为空腹血糖>=7.0",
                "score": 0.9,
            },
            {
                "node_id": "n2",
                "source": "高血压指南.md",
                "page": 12,
                "content": "高血压定义为>=140/90",
                "score": 0.85,
            },
        ]

        result = self.verifier.extract_citations_only(contexts)

        assert len(result) == 2
        assert result[0].source_id == "1"
        assert result[0].document_id == "n1"
        assert result[0].file_name == "糖尿病指南.md"
        assert result[0].page_number == 5
        assert result[0].chunk_content == "糖尿病诊断标准为空腹血糖>=7.0"
        assert result[0].relevance_score == 0.9
        assert result[0].position == CitationPosition.DIRECT
        assert result[0].verified is True
        assert result[0].quote_in_answer is None
        assert result[0].verification_message is None

    def test_missing_keys_use_defaults(self):
        contexts = [{}]

        result = self.verifier.extract_citations_only(contexts)

        assert len(result) == 1
        assert result[0].source_id == "1"
        assert result[0].document_id is None  # ctx.get("node_id") → None
        assert result[0].file_name == "未知来源"
        assert result[0].page_number is None
        assert result[0].chunk_content == ""
        assert result[0].relevance_score == 0.0

    def test_empty_contexts_list(self):
        result = self.verifier.extract_citations_only([])
        assert result == []

    def test_content_truncated_to_200_chars(self):
        long_content = "A" * 300
        contexts = [{"content": long_content, "source": "test.txt"}]

        result = self.verifier.extract_citations_only(contexts)

        assert len(result[0].chunk_content) == 200


class TestParseSourceDesc:
    """Tests for _parse_source_desc() internal method."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_with_hash_and_valid_page_number(self):
        file_name, page_number = self.verifier._parse_source_desc("糖尿病指南#5")
        assert file_name == "糖尿病指南"
        assert page_number == 5

    def test_with_hash_and_no_file_name(self):
        file_name, page_number = self.verifier._parse_source_desc("#5")
        assert file_name is None
        assert page_number == 5

    def test_without_hash(self):
        file_name, page_number = self.verifier._parse_source_desc("糖尿病指南")
        assert file_name == "糖尿病指南"
        assert page_number is None

    def test_with_hash_and_non_numeric_page(self):
        file_name, page_number = self.verifier._parse_source_desc("糖尿病指南#abc")
        assert file_name == "糖尿病指南"
        assert page_number is None

    def test_with_multiple_hashes_uses_last(self):
        file_name, page_number = self.verifier._parse_source_desc("文件#章节#10")
        assert file_name == "文件#章节"
        assert page_number == 10

    def test_empty_string(self):
        file_name, page_number = self.verifier._parse_source_desc("")
        assert file_name == ""
        assert page_number is None


class TestVerifyCitation:
    """Tests for _verify_citation() internal method."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_returns_false_when_ctx_file_empty(self):
        result = self.verifier._verify_citation(cited_file="指南.md", cited_page=1, ctx_file="", ctx_page=1)
        assert result is False

    def test_returns_true_when_cited_file_is_none(self):
        """If no cited file provided, can't verify → return True (lenient)."""
        result = self.verifier._verify_citation(cited_file=None, cited_page=None, ctx_file="指南.md", ctx_page=5)
        assert result is True

    def test_returns_false_when_files_dont_match(self):
        result = self.verifier._verify_citation(
            cited_file="高血压指南", cited_page=1, ctx_file="糖尿病指南", ctx_page=1
        )
        assert result is False

    def test_returns_false_when_page_mismatch(self):
        result = self.verifier._verify_citation(cited_file="指南", cited_page=5, ctx_file="指南", ctx_page=10)
        assert result is False

    def test_returns_true_when_all_match(self):
        result = self.verifier._verify_citation(
            cited_file="糖尿病指南", cited_page=5, ctx_file="糖尿病指南.md", ctx_page=5
        )
        assert result is True

    def test_cited_page_none_skips_page_check(self):
        result = self.verifier._verify_citation(cited_file="指南", cited_page=None, ctx_file="指南", ctx_page=10)
        assert result is True

    def test_ctx_page_none_skips_page_check(self):
        result = self.verifier._verify_citation(cited_file="指南", cited_page=5, ctx_file="指南", ctx_page=None)
        assert result is True


class TestNormalizeFilename:
    """Tests for _normalize_filename() internal method."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_removes_file_extensions(self):
        result = self.verifier._normalize_filename("糖尿病指南.md")
        assert result == "糖尿病指南"

    def test_removes_txt_extension(self):
        result = self.verifier._normalize_filename("高血压防治指南.txt")
        assert result == "高血压防治指南"

    def test_removes_page_suffix(self):
        result = self.verifier._normalize_filename("糖尿病指南#5")
        assert result == "糖尿病指南"

    def test_lowercases(self):
        result = self.verifier._normalize_filename("GUIDE.MD")
        assert result == "guide"

    def test_strips_whitespace(self):
        # strip() happens AFTER the extension regex, so the extension won't match
        # when there's trailing whitespace before the strip
        result = self.verifier._normalize_filename("  指南.md  ")
        assert result == "指南.md"

    def test_empty_string(self):
        result = self.verifier._normalize_filename("")
        assert result == ""


class TestFilesMatch:
    """Tests for _files_match() internal method."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_exact_match_after_normalization(self):
        result = self.verifier._files_match("指南", "指南")
        assert result is True

    def test_substring_match_file1_in_file2(self):
        result = self.verifier._files_match("指南", "糖尿病指南")
        assert result is True

    def test_substring_match_file2_in_file1(self):
        result = self.verifier._files_match("糖尿病诊疗指南", "诊疗指南")
        assert result is True

    def test_medical_suffix_normalization_match(self):
        """After removing common medical suffixes, the base names match."""
        result = self.verifier._files_match("糖尿病诊疗指南", "糖尿病诊疗共识")
        assert result is True

    def test_no_match_different_files(self):
        result = self.verifier._files_match("糖尿病", "高血压")
        assert result is False

    def test_empty_file1_returns_false(self):
        result = self.verifier._files_match("", "指南")
        assert result is False

    def test_empty_file2_returns_false(self):
        result = self.verifier._files_match("指南", "")
        assert result is False

    def test_both_empty_returns_false(self):
        result = self.verifier._files_match("", "")
        assert result is False


class TestCitationResultShape:
    """Tests for the shape and completeness of returned Citation objects."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_citation_has_all_required_fields(self):
        answer = "根据[来源1](指南#3)所述。"
        contexts = [_make_ctx(idx="1", source_file="指南.md", page_number=3)]

        result = self.verifier.extract_and_verify(answer, contexts)

        for citation in result:
            assert isinstance(citation, Citation)
            assert isinstance(citation.source_id, str)
            assert isinstance(citation.file_name, str)
            assert isinstance(citation.relevance_score, float)
            assert isinstance(citation.verified, bool)
            assert isinstance(citation.position, CitationPosition)
            assert hasattr(citation, "document_id")
            assert hasattr(citation, "page_number")
            assert hasattr(citation, "chunk_content")
            assert hasattr(citation, "quote_in_answer")
            assert hasattr(citation, "verification_message")

    def test_direct_citation_includes_chunk_content(self):
        answer = "根据[来源1](指南#3)所述。"
        contexts = [_make_ctx(idx="1", content="全文内容包括重要医学信息" * 5, page_number=3)]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.position == CitationPosition.DIRECT]
        assert len(direct) == 1
        assert len(direct[0].chunk_content) > 0
        assert len(direct[0].chunk_content) <= 200

    def test_direct_citation_has_quote_in_answer(self):
        answer = "根据[来源1](指南#3)所述。"
        contexts = [_make_ctx(idx="1", page_number=3)]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.position == CitationPosition.DIRECT]
        assert len(direct) == 1
        assert direct[0].quote_in_answer is not None
        assert direct[0].quote_in_answer == "[来源1](指南#3)"

    def test_verified_citation_has_no_verification_message(self):
        answer = "[来源1](指南#3)"
        contexts = [_make_ctx(idx="1", source_file="指南.md", page_number=3)]

        result = self.verifier.extract_and_verify(answer, contexts)

        direct = [c for c in result if c.position == CitationPosition.DIRECT]
        assert len(direct) == 1
        assert direct[0].verified is True
        assert direct[0].verification_message is None

    def test_unverified_citation_has_verification_message(self):
        answer = "[来源1](高血压指南#3)"
        contexts = [_make_ctx(idx="1", source_file="糖尿病指南.md")]

        result = self.verifier.extract_and_verify(answer, contexts)

        unverified = [c for c in result if c.position == CitationPosition.UNVERIFIED]
        assert len(unverified) == 1
        assert unverified[0].verified is False
        assert unverified[0].verification_message is not None


class TestMetadataFallback:
    """Tests for metadata fallback when source description is empty."""

    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_empty_source_desc_uses_metadata(self):
        answer = "参考[来源1](糖尿病诊疗指南#5)"
        contexts = [
            _make_ctx(
                idx="1",
                source_file="糖尿病诊疗指南.md",
                page_number=5,
                doc_id="doc-abc",
            )
        ]

        result = self.verifier.extract_and_verify(answer, contexts)

        assert len(result) == 1
        assert result[0].document_id == "doc-abc"
        assert "糖尿病诊疗指南" in result[0].file_name

    def test_metadata_missing_source_file(self):
        answer = "参考[来源1](糖尿病诊疗指南#5)"
        ctx = RetrievedNode(
            node_id="1",
            content="test",
            score=0.9,
            metadata={"doc_id": "doc-1"},
        )
        result = self.verifier.extract_and_verify(answer, [ctx])

        # ctx metadata has no source_file → _verify_citation returns False (empty ctx_file)
        # → position=UNVERIFIED
        unverified = [c for c in result if c.position == CitationPosition.UNVERIFIED]
        assert len(unverified) == 1
        assert unverified[0].verified is False
