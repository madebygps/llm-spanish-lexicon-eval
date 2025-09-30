"""Tests for data_loader module."""

import json

from data_loader import load_models, load_prompts, load_vocabulary


class TestLoadModels:
    """Tests for load_models function."""
    
    def test_load_models_filters_comments(self, tmp_path, monkeypatch):
        """Test that commented models are excluded"""
        models_file = tmp_path / "suite" / "models_list.txt"
        models_file.parent.mkdir(parents=True)
        models_file.write_text(
            "#solar:10.7b\n"
            "gemma3:12b\n"
            "#mistral:latest\n"
            "llama3.1:latest\n"
        )
        
        monkeypatch.chdir(tmp_path)
        models = load_models()
        
        assert len(models) == 2
        assert "gemma3:12b" in models
        assert "llama3.1:latest" in models
        assert "#solar:10.7b" not in models
        assert "#mistral:latest" not in models
    
    def test_load_models_returns_list(self, tmp_path, monkeypatch):
        """Test that load_models returns a list"""
        models_file = tmp_path / "suite" / "models_list.txt"
        models_file.parent.mkdir(parents=True)
        models_file.write_text("model1\nmodel2\n")
        
        monkeypatch.chdir(tmp_path)
        models = load_models()
        
        assert isinstance(models, list)
        assert len(models) == 2
    
    def test_load_models_excludes_empty_lines(self, tmp_path, monkeypatch):
        """Test that empty lines are excluded"""
        models_file = tmp_path / "suite" / "models_list.txt"
        models_file.parent.mkdir(parents=True)
        models_file.write_text(
            "model1\n"
            "\n"
            "model2\n"
            "   \n"
            "model3\n"
        )
        
        monkeypatch.chdir(tmp_path)
        models = load_models()
        
        assert len(models) == 3
        assert "model1" in models
        assert "model2" in models
        assert "model3" in models
    
    def test_load_models_strips_whitespace(self, tmp_path, monkeypatch):
        """Test that whitespace is stripped from model names"""
        models_file = tmp_path / "suite" / "models_list.txt"
        models_file.parent.mkdir(parents=True)
        models_file.write_text(
            "  model1  \n"
            "model2\n"
            "   model3\n"
        )
        
        monkeypatch.chdir(tmp_path)
        models = load_models()
        
        assert "model1" in models
        assert "model2" in models
        assert "model3" in models
        assert "  model1  " not in models


class TestLoadPrompts:
    """Tests for load_prompts function."""
    
    def test_load_prompts_returns_dict(self, tmp_path, monkeypatch):
        """Test that load_prompts returns a dictionary"""
        prompts_file = tmp_path / "suite" / "prompts.json"
        prompts_file.parent.mkdir(parents=True)
        prompts_data = {
            "prompt_a": "Define {word}",
            "prompt_b": "Use {word} in a sentence"
        }
        prompts_file.write_text(json.dumps(prompts_data))
        
        monkeypatch.chdir(tmp_path)
        prompts = load_prompts()
        
        assert isinstance(prompts, dict)
        assert "prompt_a" in prompts
        assert "prompt_b" in prompts
    
    def test_load_prompts_contains_word_placeholder(self, tmp_path, monkeypatch):
        """Test that prompts contain {word} placeholder"""
        prompts_file = tmp_path / "suite" / "prompts.json"
        prompts_file.parent.mkdir(parents=True)
        prompts_data = {
            "prompt_a": "Define the word {word}",
            "prompt_b": "Use {word} in a sentence"
        }
        prompts_file.write_text(json.dumps(prompts_data))
        
        monkeypatch.chdir(tmp_path)
        prompts = load_prompts()
        
        assert "{word}" in prompts["prompt_a"]
        assert "{word}" in prompts["prompt_b"]
    
    def test_load_prompts_preserves_utf8(self, tmp_path, monkeypatch):
        """Test that UTF-8 characters are preserved"""
        prompts_file = tmp_path / "suite" / "prompts.json"
        prompts_file.parent.mkdir(parents=True)
        prompts_data = {
            "prompt_a": "Define la palabra {word} en español",
            "prompt_b": "Usa {word} en una oración"
        }
        prompts_file.write_text(json.dumps(prompts_data, ensure_ascii=False))
        
        monkeypatch.chdir(tmp_path)
        prompts = load_prompts()
        
        assert "español" in prompts["prompt_a"]
        assert "oración" in prompts["prompt_b"]


class TestLoadVocabulary:
    """Tests for load_vocabulary function."""
    
    def test_load_vocabulary_returns_list(self, tmp_path, monkeypatch):
        """Test that load_vocabulary returns a list"""
        vocab_file = tmp_path / "suite" / "vocabulary_short.json"
        vocab_file.parent.mkdir(parents=True)
        vocab_data = [
            {"word": "ardilla", "answer": "A squirrel"},
            {"word": "corbata", "answer": "A necktie"}
        ]
        vocab_file.write_text(json.dumps(vocab_data))
        
        monkeypatch.chdir(tmp_path)
        vocabulary = load_vocabulary()
        
        assert isinstance(vocabulary, list)
        assert len(vocabulary) == 2
    
    def test_load_vocabulary_contains_word_and_answer(self, tmp_path, monkeypatch):
        """Test that vocabulary entries have word and answer fields"""
        vocab_file = tmp_path / "suite" / "vocabulary_short.json"
        vocab_file.parent.mkdir(parents=True)
        vocab_data = [
            {"word": "ardilla", "answer": "A squirrel"}
        ]
        vocab_file.write_text(json.dumps(vocab_data))
        
        monkeypatch.chdir(tmp_path)
        vocabulary = load_vocabulary()
        
        assert "word" in vocabulary[0]
        assert "answer" in vocabulary[0]
        assert vocabulary[0]["word"] == "ardilla"
        assert vocabulary[0]["answer"] == "A squirrel"
    
    def test_load_vocabulary_preserves_utf8(self, tmp_path, monkeypatch):
        """Test that UTF-8 characters are preserved"""
        vocab_file = tmp_path / "suite" / "vocabulary_short.json"
        vocab_file.parent.mkdir(parents=True)
        vocab_data = [
            {"word": "agüista", "answer": "Persona que toma aguas medicinales"}
        ]
        vocab_file.write_text(json.dumps(vocab_data, ensure_ascii=False))
        
        monkeypatch.chdir(tmp_path)
        vocabulary = load_vocabulary()
        
        assert vocabulary[0]["word"] == "agüista"
        assert "medicinales" in vocabulary[0]["answer"]
