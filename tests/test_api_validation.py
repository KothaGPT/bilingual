import warnings

import pytest

from bilingual import api


def test_translate_invalid_source_lang_raises_value_error() -> None:
    with pytest.raises(ValueError):
        api.translate("text", src="xx", tgt="en")


def test_translate_same_lang_warns_and_returns_original() -> None:
    text = "Hello"
    with pytest.warns(UserWarning):
        result = api.translate(text, src="en", tgt="en")
    assert result == text


def test_tokenize_with_invalid_tokenizer_type_raises_type_error() -> None:
    with pytest.raises(TypeError):
        api.tokenize("text", tokenizer=object())


def test_classify_with_empty_labels_raises_value_error() -> None:
    with pytest.raises(ValueError):
        api.classify("some text", labels=[])


def test_safety_check_with_invalid_lang_warns_and_autodetects() -> None:
    with pytest.warns(UserWarning):
        result = api.safety_check("This is a safe sentence.", lang="xx")
    assert result["language"] in {"bn", "en"}


def test_readability_check_with_invalid_lang_warns_and_autodetects() -> None:
    with pytest.warns(UserWarning):
        result = api.readability_check("This is a simple sentence.", lang="xx")
    assert result["language"] in {"bn", "en"}


def test_batch_process_with_non_list_texts_raises_type_error() -> None:
    with pytest.raises(TypeError):
        api.batch_process("not a list", operation="tokenize")


def test_batch_process_classify_propagates_empty_labels_error() -> None:
    with pytest.raises(ValueError):
        api.batch_process(["text"], operation="classify", labels=[])
