"""
Tests for Named Entity Recognizer.
"""

import pytest
import torch

from bilingual.models.named_entity_recognizer import BanglaNER


class TestBanglaNER:
    """Test suite for Bangla NER."""

    @pytest.fixture
    def ner(self):
        """Create a test NER instance."""
        return BanglaNER()

    def test_ner_initialization(self, ner):
        """Test NER initialization."""
        assert ner is not None
        assert ner.device in ["cuda", "cpu"]
        assert len(ner.LABEL_NAMES) == 19
        assert len(ner.ENTITY_TYPES) == 9

    def test_label_mappings(self, ner):
        """Test label ID mappings."""
        assert ner.id2label[0] == "O"
        assert ner.id2label[1] == "B-PER"
        assert ner.label2id["O"] == 0
        assert ner.label2id["B-PER"] == 1

    def test_entity_types(self, ner):
        """Test entity type definitions."""
        expected_types = ["PER", "ORG", "LOC", "DATE", "TIME", "WORK", "EVENT", "LANG", "MISC"]
        assert ner.ENTITY_TYPES == expected_types

    def test_recognize_without_model(self, ner):
        """Test recognition without pretrained model."""
        if ner.model is None:
            with pytest.raises(ValueError, match="Model not initialized"):
                ner.recognize("রবীন্দ্রনাথ ঠাকুর শান্তিনিকেতনে থাকতেন")

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_recognize_person(self, ner):
        """Test person entity recognition."""
        text = "রবীন্দ্রনাথ ঠাকুর একজন মহান কবি ছিলেন"
        result = ner.recognize(text)

        assert "entities" in result
        assert "entities_by_type" in result
        assert "PER" in result["entities_by_type"]

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_recognize_organization(self, ner):
        """Test organization entity recognition."""
        text = "বিশ্বভারতী বিশ্ববিদ্যালয় শান্তিনিকেতনে অবস্থিত"
        result = ner.recognize(text)

        assert "ORG" in result["entities_by_type"]

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_recognize_location(self, ner):
        """Test location entity recognition."""
        text = "ঢাকা বাংলাদেশের রাজধানী"
        result = ner.recognize(text)

        assert "LOC" in result["entities_by_type"]

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_recognize_literary_work(self, ner):
        """Test literary work entity recognition."""
        text = "গীতাঞ্জলি রবীন্দ্রনাথের বিখ্যাত কাব্যগ্রন্থ"
        result = ner.recognize(text)

        assert "WORK" in result["entities_by_type"]

    def test_batch_recognize(self, ner):
        """Test batch recognition."""
        if ner.model is None:
            pytest.skip("Model not available")

        texts = [
            "রবীন্দ্রনাথ ঠাকুর",
            "কাজী নজরুল ইসলাম",
        ]

        results = ner.batch_recognize(texts)
        assert len(results) == 2

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_get_entity_statistics(self, ner):
        """Test entity statistics."""
        text = "রবীন্দ্রনাথ ঠাকুর শান্তিনিকেতনে বিশ্বভারতী প্রতিষ্ঠা করেন"
        stats = ner.get_entity_statistics(text)

        assert "total_entities" in stats
        assert "entity_counts" in stats
        assert "entity_density" in stats
        assert "most_common_type" in stats

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_extract_literary_references(self, ner):
        """Test literary reference extraction."""
        text = "রবীন্দ্রনাথ ঠাকুর গীতাঞ্জলি রচনা করেন এবং নোবেল পুরস্কার পান"
        refs = ner.extract_literary_references(text)

        assert "authors" in refs
        assert "works" in refs
        assert "events" in refs
        assert "locations" in refs

    def test_extract_entities_empty(self, ner):
        """Test entity extraction with no entities."""
        entities = ner._extract_entities(
            text="একটি সাধারণ বাক্য",
            tokens=["একটি", "সাধারণ", "বাক্য"],
            labels=["O", "O", "O"],
            confidences=[0.9, 0.9, 0.9],
            offset_mapping=torch.tensor([[0, 4], [5, 11], [12, 16]]),
            entity_type="PER",
        )

        assert len(entities) == 0
