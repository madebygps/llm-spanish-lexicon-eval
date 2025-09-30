"""Tests for evaluator module."""

import pytest

from evaluator import calculate_accuracy
from storage import save_response


class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""
    
    @pytest.fixture
    def sample_vocabulary(self):
        """Sample vocabulary for testing"""
        return [
            {"word": "word1", "answer": "Definition 1"},
            {"word": "word2", "answer": "Definition 2"},
            {"word": "word3", "answer": "Definition 3"},
        ]
    
    def test_calculate_accuracy_all_correct_prompt_a(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test accuracy calculation when all prompt A responses are correct"""
        monkeypatch.chdir(tmp_path)
        
        model = "test-model"
        for entry in sample_vocabulary:
            save_response(
                model, 
                entry["word"], 
                entry["answer"],
                model_response_a="response",
                judgment_a="correct"
            )
        
        accuracy = calculate_accuracy(model, sample_vocabulary, "a")
        assert accuracy == 100.0
    
    def test_calculate_accuracy_all_incorrect_prompt_a(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test accuracy calculation when all prompt A responses are incorrect"""
        monkeypatch.chdir(tmp_path)
        
        model = "test-model"
        for entry in sample_vocabulary:
            save_response(
                model,
                entry["word"],
                entry["answer"],
                model_response_a="response",
                judgment_a="incorrect"
            )
        
        accuracy = calculate_accuracy(model, sample_vocabulary, "a")
        assert accuracy == 0.0
    
    def test_calculate_accuracy_partial_correct_prompt_a(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test accuracy calculation with some correct prompt A responses"""
        monkeypatch.chdir(tmp_path)
        
        model = "test-model"
        save_response(model, "word1", "def1", model_response_a="resp", judgment_a="correct")
        save_response(model, "word2", "def2", model_response_a="resp", judgment_a="incorrect")
        save_response(model, "word3", "def3", model_response_a="resp", judgment_a="correct")
        
        accuracy = calculate_accuracy(model, sample_vocabulary, "a")
        assert accuracy == pytest.approx(66.67, rel=0.1)
    
    def test_calculate_accuracy_all_correct_prompt_b(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test accuracy calculation when all prompt B responses are correct"""
        monkeypatch.chdir(tmp_path)
        
        model = "test-model"
        for entry in sample_vocabulary:
            save_response(
                model,
                entry["word"],
                entry["answer"],
                model_response_b="response",
                judgment_b="correct"
            )
        
        accuracy = calculate_accuracy(model, sample_vocabulary, "b")
        assert accuracy == 100.0
    
    def test_calculate_accuracy_partial_correct_prompt_b(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test accuracy calculation with some correct prompt B responses"""
        monkeypatch.chdir(tmp_path)
        
        model = "test-model"
        save_response(model, "word1", "def1", model_response_b="resp", judgment_b="correct")
        save_response(model, "word2", "def2", model_response_b="resp", judgment_b="correct")
        save_response(model, "word3", "def3", model_response_b="resp", judgment_b="incorrect")
        
        accuracy = calculate_accuracy(model, sample_vocabulary, "b")
        assert accuracy == pytest.approx(66.67, rel=0.1)
    
    def test_calculate_accuracy_empty_vocabulary(self, tmp_path, monkeypatch):
        """Test accuracy calculation with empty vocabulary"""
        monkeypatch.chdir(tmp_path)
        
        accuracy = calculate_accuracy("test-model", [], "a")
        assert accuracy == 0.0
    
    def test_calculate_accuracy_missing_judgment(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test accuracy when some judgments are missing"""
        monkeypatch.chdir(tmp_path)
        
        model = "test-model"
        # Only save response for word1, no judgment
        save_response(model, "word1", "def1", model_response_a="resp")
        save_response(model, "word2", "def2", model_response_a="resp", judgment_a="correct")
        save_response(model, "word3", "def3", model_response_a="resp", judgment_a="correct")
        
        accuracy = calculate_accuracy(model, sample_vocabulary, "a")
        # Only 2 out of 3 are correct (word1 has no judgment)
        assert accuracy == pytest.approx(66.67, rel=0.1)
    
    def test_calculate_accuracy_no_responses(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test accuracy when no responses exist"""
        monkeypatch.chdir(tmp_path)
        
        # Don't save any responses
        accuracy = calculate_accuracy("test-model", sample_vocabulary, "a")
        assert accuracy == 0.0
    
    def test_calculate_accuracy_single_word(self, tmp_path, monkeypatch):
        """Test accuracy with single word vocabulary"""
        monkeypatch.chdir(tmp_path)
        
        vocabulary = [{"word": "test", "answer": "definition"}]
        save_response("model", "test", "definition", model_response_a="resp", judgment_a="correct")
        
        accuracy = calculate_accuracy("model", vocabulary, "a")
        assert accuracy == 100.0
    
    def test_calculate_accuracy_default_prompt_type(self, tmp_path, monkeypatch, sample_vocabulary):
        """Test that default prompt type is 'a'"""
        monkeypatch.chdir(tmp_path)
        
        model = "test-model"
        for entry in sample_vocabulary:
            save_response(
                model,
                entry["word"],
                entry["answer"],
                model_response_a="response",
                judgment_a="correct"
            )
        
        # Call without prompt_type parameter
        accuracy = calculate_accuracy(model, sample_vocabulary)
        assert accuracy == 100.0
