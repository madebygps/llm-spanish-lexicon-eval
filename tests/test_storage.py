"""Tests for storage module."""

import json

from storage import save_response, load_response, update_response_judgment


class TestSaveResponse:
    """Tests for save_response function."""
    
    def test_save_response_creates_json_file(self, tmp_path, monkeypatch):
        """Test that save_response creates a JSON file"""
        monkeypatch.chdir(tmp_path)
        
        save_response("test-model", "ardilla", "A squirrel", model_response_a="Es un roedor")
        
        file_path = tmp_path / "output" / "test-model" / "ardilla.json"
        assert file_path.exists()
    
    def test_save_response_has_correct_structure(self, tmp_path, monkeypatch):
        """Test that saved response has correct JSON structure"""
        monkeypatch.chdir(tmp_path)
        
        save_response("test-model", "ardilla", "A squirrel", model_response_a="Es un roedor")
        
        file_path = tmp_path / "output" / "test-model" / "ardilla.json"
        with open(file_path) as f:
            data = json.load(f)
        
        assert data["word"] == "ardilla"
        assert data["correct_definition"] == "A squirrel"
        assert data["model_response_a"] == "Es un roedor"
    
    def test_save_response_creates_directory_if_not_exists(self, tmp_path, monkeypatch):
        """Test that output directory is created if it doesn't exist"""
        monkeypatch.chdir(tmp_path)
        
        output_dir = tmp_path / "output" / "new-model"
        assert not output_dir.exists()
        
        save_response("new-model", "word", "definition", model_response_a="response")
        
        assert output_dir.exists()
    
    def test_save_response_preserves_existing_fields(self, tmp_path, monkeypatch):
        """Test that existing fields are preserved when updating"""
        monkeypatch.chdir(tmp_path)
        
        # First save with response_a
        save_response("model", "word", "def", model_response_a="response_a")
        
        # Then save with response_b
        save_response("model", "word", "def", model_response_b="response_b")
        
        # Verify both are present
        data = load_response("model", "word")
        assert data["model_response_a"] == "response_a"
        assert data["model_response_b"] == "response_b"
    
    def test_save_response_preserves_utf8(self, tmp_path, monkeypatch):
        """Test that UTF-8 characters are preserved"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "agüista", "Persona que toma aguas", model_response_a="Definición")
        
        file_path = tmp_path / "output" / "model" / "agüista.json"
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["word"] == "agüista"
        assert data["model_response_a"] == "Definición"
    
    def test_save_response_with_both_responses(self, tmp_path, monkeypatch):
        """Test saving both responses at once"""
        monkeypatch.chdir(tmp_path)
        
        save_response(
            "model", 
            "word", 
            "def", 
            model_response_a="response_a",
            model_response_b="response_b"
        )
        
        data = load_response("model", "word")
        assert data["model_response_a"] == "response_a"
        assert data["model_response_b"] == "response_b"
    
    def test_save_response_with_judgments(self, tmp_path, monkeypatch):
        """Test saving with judgment fields"""
        monkeypatch.chdir(tmp_path)
        
        save_response(
            "model", 
            "word", 
            "def", 
            model_response_a="response",
            judgment_a="correct"
        )
        
        data = load_response("model", "word")
        assert data["judgment_a"] == "correct"
    
    def test_save_response_empty_strings_not_saved(self, tmp_path, monkeypatch):
        """Test that empty string parameters are not saved"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "word", "def", model_response_a="", model_response_b="")
        
        data = load_response("model", "word")
        assert "model_response_a" not in data
        assert "model_response_b" not in data


class TestLoadResponse:
    """Tests for load_response function."""
    
    def test_load_response_returns_dict(self, tmp_path, monkeypatch):
        """Test that load_response returns a dictionary"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "word", "def", model_response_a="response")
        result = load_response("model", "word")
        
        assert isinstance(result, dict)
    
    def test_load_response_returns_empty_dict_when_not_exists(self, tmp_path, monkeypatch):
        """Test that load_response returns empty dict for non-existent file"""
        monkeypatch.chdir(tmp_path)
        
        result = load_response("nonexistent-model", "palabra")
        
        assert result == {}
    
    def test_load_response_returns_saved_data(self, tmp_path, monkeypatch):
        """Test that load_response returns previously saved data"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "ardilla", "squirrel", model_response_a="rodent")
        result = load_response("model", "ardilla")
        
        assert result["word"] == "ardilla"
        assert result["correct_definition"] == "squirrel"
        assert result["model_response_a"] == "rodent"
    
    def test_load_response_with_utf8(self, tmp_path, monkeypatch):
        """Test loading response with UTF-8 characters"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "agüista", "definición", model_response_a="respuesta")
        result = load_response("model", "agüista")
        
        assert result["word"] == "agüista"
        assert result["correct_definition"] == "definición"


class TestUpdateResponseJudgment:
    """Tests for update_response_judgment function."""
    
    def test_update_response_judgment_adds_judgment_a(self, tmp_path, monkeypatch):
        """Test that update_response_judgment adds judgment_a"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "word", "def", model_response_a="response")
        update_response_judgment("model", "word", judgment_a="correct")
        
        data = load_response("model", "word")
        assert data["judgment_a"] == "correct"
    
    def test_update_response_judgment_adds_judgment_b(self, tmp_path, monkeypatch):
        """Test that update_response_judgment adds judgment_b"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "word", "def", model_response_b="response")
        update_response_judgment("model", "word", judgment_b="incorrect")
        
        data = load_response("model", "word")
        assert data["judgment_b"] == "incorrect"
    
    def test_update_response_judgment_preserves_existing_data(self, tmp_path, monkeypatch):
        """Test that existing data is preserved when updating judgment"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "word", "def", model_response_a="response_a")
        update_response_judgment("model", "word", judgment_a="correct")
        
        data = load_response("model", "word")
        assert data["model_response_a"] == "response_a"
        assert data["judgment_a"] == "correct"
    
    def test_update_response_judgment_updates_both(self, tmp_path, monkeypatch):
        """Test updating both judgments at once"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "word", "def", 
                     model_response_a="resp_a", 
                     model_response_b="resp_b")
        update_response_judgment("model", "word", 
                                judgment_a="correct", 
                                judgment_b="incorrect")
        
        data = load_response("model", "word")
        assert data["judgment_a"] == "correct"
        assert data["judgment_b"] == "incorrect"
    
    def test_update_response_judgment_overwrites_existing(self, tmp_path, monkeypatch):
        """Test that existing judgment is overwritten"""
        monkeypatch.chdir(tmp_path)
        
        save_response("model", "word", "def", judgment_a="incorrect")
        update_response_judgment("model", "word", judgment_a="correct")
        
        data = load_response("model", "word")
        assert data["judgment_a"] == "correct"
