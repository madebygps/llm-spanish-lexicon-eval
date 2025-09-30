"""Tests for reporter module."""

import json
from unittest.mock import patch, Mock

from reporter import generate_summary
from storage import save_response


class TestGenerateSummary:
    """Tests for generate_summary function."""
    
    def test_generate_summary_creates_json_file(self, tmp_path, monkeypatch):
        """Test that generate_summary creates summary.json file"""
        monkeypatch.chdir(tmp_path)
        
        models = ["model1"]
        vocabulary = [{"word": "word1", "answer": "def1"}]
        
        save_response("model1", "word1", "def1", 
                     model_response_a="resp", judgment_a="correct",
                     model_response_b="resp", judgment_b="correct")
        
        generate_summary(models, vocabulary)
        
        summary_file = tmp_path / "summary.json"
        assert summary_file.exists()
    
    def test_generate_summary_correct_structure(self, tmp_path, monkeypatch):
        """Test that summary.json has correct structure"""
        monkeypatch.chdir(tmp_path)
        
        models = ["model1"]
        vocabulary = [{"word": "word1", "answer": "def1"}]
        
        save_response("model1", "word1", "def1",
                     model_response_a="resp", judgment_a="correct",
                     model_response_b="resp", judgment_b="incorrect")
        
        generate_summary(models, vocabulary)
        
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        
        assert "model1" in summary
        assert "prompt_a_accuracy" in summary["model1"]
        assert "prompt_b_accuracy" in summary["model1"]
    
    def test_generate_summary_calculates_accuracy_correctly(self, tmp_path, monkeypatch):
        """Test that accuracy is calculated correctly"""
        monkeypatch.chdir(tmp_path)
        
        models = ["model1"]
        vocabulary = [
            {"word": "word1", "answer": "def1"},
            {"word": "word2", "answer": "def2"},
        ]
        
        save_response("model1", "word1", "def1",
                     model_response_a="resp", judgment_a="correct",
                     model_response_b="resp", judgment_b="correct")
        save_response("model1", "word2", "def2",
                     model_response_a="resp", judgment_a="incorrect",
                     model_response_b="resp", judgment_b="correct")
        
        generate_summary(models, vocabulary)
        
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        
        assert summary["model1"]["prompt_a_accuracy"] == 50.0
        assert summary["model1"]["prompt_b_accuracy"] == 100.0
    
    def test_generate_summary_multiple_models(self, tmp_path, monkeypatch):
        """Test summary generation for multiple models"""
        monkeypatch.chdir(tmp_path)
        
        models = ["model1", "model2"]
        vocabulary = [{"word": "word1", "answer": "def1"}]
        
        save_response("model1", "word1", "def1",
                     model_response_a="resp", judgment_a="correct",
                     model_response_b="resp", judgment_b="correct")
        save_response("model2", "word1", "def1",
                     model_response_a="resp", judgment_a="incorrect",
                     model_response_b="resp", judgment_b="incorrect")
        
        generate_summary(models, vocabulary)
        
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        
        assert "model1" in summary
        assert "model2" in summary
        assert summary["model1"]["prompt_a_accuracy"] == 100.0
        assert summary["model2"]["prompt_a_accuracy"] == 0.0
    
    def test_generate_summary_preserves_utf8(self, tmp_path, monkeypatch):
        """Test that UTF-8 characters are preserved in summary"""
        monkeypatch.chdir(tmp_path)
        
        models = ["modelo-español"]
        vocabulary = [{"word": "agüista", "answer": "definición"}]
        
        save_response("modelo-español", "agüista", "definición",
                     model_response_a="respuesta", judgment_a="correct",
                     model_response_b="respuesta", judgment_b="correct")
        
        generate_summary(models, vocabulary)
        
        with open(tmp_path / "summary.json", encoding='utf-8') as f:
            summary = json.load(f)
        
        assert "modelo-español" in summary
    
    @patch('reporter.Console')
    def test_generate_summary_displays_table(self, mock_console_class, tmp_path, monkeypatch):
        """Test that summary displays a table"""
        monkeypatch.chdir(tmp_path)
        
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        models = ["model1"]
        vocabulary = [{"word": "word1", "answer": "def1"}]
        
        save_response("model1", "word1", "def1",
                     model_response_a="resp", judgment_a="correct",
                     model_response_b="resp", judgment_b="correct")
        
        generate_summary(models, vocabulary)
        
        # Verify console.print was called (to display table)
        mock_console.print.assert_called_once()
    
    def test_generate_summary_empty_vocabulary(self, tmp_path, monkeypatch):
        """Test summary generation with empty vocabulary"""
        monkeypatch.chdir(tmp_path)
        
        models = ["model1"]
        vocabulary = []
        
        generate_summary(models, vocabulary)
        
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        
        assert summary["model1"]["prompt_a_accuracy"] == 0.0
        assert summary["model1"]["prompt_b_accuracy"] == 0.0
    
    def test_generate_summary_no_judgments(self, tmp_path, monkeypatch):
        """Test summary when responses exist but no judgments"""
        monkeypatch.chdir(tmp_path)
        
        models = ["model1"]
        vocabulary = [{"word": "word1", "answer": "def1"}]
        
        # Save response without judgment
        save_response("model1", "word1", "def1",
                     model_response_a="resp",
                     model_response_b="resp")
        
        generate_summary(models, vocabulary)
        
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        
        assert summary["model1"]["prompt_a_accuracy"] == 0.0
        assert summary["model1"]["prompt_b_accuracy"] == 0.0
