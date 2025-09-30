"""Tests for model_client module."""

from unittest.mock import Mock, patch

from model_client import prompt_model, judge_response, judge_response_b


class TestPromptModel:
    """Tests for prompt_model function."""
    
    @patch('model_client.OpenAI')
    def test_prompt_model_calls_ollama_api(self, mock_openai_class):
        """Test that prompt_model correctly calls Ollama API"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Una ardilla es un roedor"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = prompt_model("ardilla", "llama3.1:latest", "Define {word}")
        
        assert result == "Una ardilla es un roedor"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('model_client.OpenAI')
    def test_prompt_model_uses_correct_base_url(self, mock_openai_class):
        """Test that prompt_model uses Ollama base URL"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_response
        
        prompt_model("word", "model", "template {word}")
        
        # Verify OpenAI was initialized with Ollama URL
        mock_openai_class.assert_called_once_with(
            base_url='http://localhost:11434/v1/',
            api_key='ollama'
        )
    
    @patch('model_client.OpenAI')
    def test_prompt_model_formats_word_in_template(self, mock_openai_class):
        """Test that {word} placeholder is replaced in template"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_response
        
        prompt_model("ardilla", "model", "Define {word} in Spanish")
        
        # Verify the formatted prompt was sent
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        assert "Define ardilla in Spanish" in messages[0]['content']
    
    @patch('model_client.OpenAI')
    def test_prompt_model_uses_correct_model(self, mock_openai_class):
        """Test that the correct model name is used"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_response
        
        prompt_model("word", "gemma3:12b", "template {word}")
        
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == "gemma3:12b"
    
    @patch('model_client.OpenAI')
    def test_prompt_model_handles_none_response(self, mock_openai_class):
        """Test that None response is converted to empty string"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        
        result = prompt_model("word", "model", "template {word}")
        
        assert result == ""
    
    @patch('model_client.OpenAI')
    def test_prompt_model_with_utf8_word(self, mock_openai_class):
        """Test prompt_model with UTF-8 word"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Definición con acentos"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = prompt_model("agüista", "model", "Define {word}")
        
        assert result == "Definición con acentos"
        call_args = mock_client.chat.completions.create.call_args
        assert "agüista" in call_args.kwargs['messages'][0]['content']


class TestJudgeResponse:
    """Tests for judge_response function."""
    
    @patch('model_client.OpenAI')
    def test_judge_response_returns_correct(self, mock_openai_class):
        """Test that judge_response returns 'correct' for good definition"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "correct"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response("word", "definition", "good response")
        
        assert result == "correct"
    
    @patch('model_client.OpenAI')
    def test_judge_response_returns_incorrect(self, mock_openai_class):
        """Test that judge_response returns 'incorrect' for bad definition"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "incorrect"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response("word", "definition", "bad response")
        
        assert result == "incorrect"
    
    @patch('model_client.OpenAI')
    def test_judge_response_uses_gpt5(self, mock_openai_class):
        """Test that judge_response uses GPT-5 model"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "correct"
        mock_client.chat.completions.create.return_value = mock_response
        
        judge_response("word", "definition", "response")
        
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == "gpt-5"
    
    @patch('model_client.OpenAI')
    def test_judge_response_strips_whitespace(self, mock_openai_class):
        """Test that judge_response strips whitespace from response"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "  correct  "
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response("word", "definition", "response")
        
        assert result == "correct"
    
    @patch('model_client.OpenAI')
    def test_judge_response_converts_to_lowercase(self, mock_openai_class):
        """Test that judge_response converts to lowercase"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "CORRECT"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response("word", "definition", "response")
        
        assert result == "correct"
    
    @patch('model_client.OpenAI')
    def test_judge_response_handles_none_response(self, mock_openai_class):
        """Test that judge_response handles None response"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response("word", "definition", "response")
        
        assert result == "incorrect"
    
    @patch('model_client.OpenAI')
    def test_judge_response_includes_word_in_prompt(self, mock_openai_class):
        """Test that judge prompt includes the word"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "correct"
        mock_client.chat.completions.create.return_value = mock_response
        
        judge_response("ardilla", "squirrel", "response")
        
        call_args = mock_client.chat.completions.create.call_args
        prompt_content = call_args.kwargs['messages'][0]['content']
        assert "ardilla" in prompt_content.lower()


class TestJudgeResponseB:
    """Tests for judge_response_b function."""
    
    @patch('model_client.OpenAI')
    def test_judge_response_b_returns_correct(self, mock_openai_class):
        """Test that judge_response_b returns 'correct' for valid sentences"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "correct"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response_b("word", "definition", "good sentences")
        
        assert result == "correct"
    
    @patch('model_client.OpenAI')
    def test_judge_response_b_returns_incorrect(self, mock_openai_class):
        """Test that judge_response_b returns 'incorrect' for bad sentences"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "incorrect"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response_b("word", "definition", "bad sentences")
        
        assert result == "incorrect"
    
    @patch('model_client.OpenAI')
    def test_judge_response_b_uses_gpt5(self, mock_openai_class):
        """Test that judge_response_b uses GPT-5 model"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "correct"
        mock_client.chat.completions.create.return_value = mock_response
        
        judge_response_b("word", "definition", "response")
        
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == "gpt-5"
    
    @patch('model_client.OpenAI')
    def test_judge_response_b_strips_and_lowercases(self, mock_openai_class):
        """Test that judge_response_b strips whitespace and converts to lowercase"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "  INCORRECT  "
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response_b("word", "definition", "response")
        
        assert result == "incorrect"
    
    @patch('model_client.OpenAI')
    def test_judge_response_b_handles_none_response(self, mock_openai_class):
        """Test that judge_response_b handles None response"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        
        result = judge_response_b("word", "definition", "response")
        
        assert result == "incorrect"
    
    @patch('model_client.OpenAI')
    def test_judge_response_b_includes_word_in_prompt(self, mock_openai_class):
        """Test that judge_response_b prompt includes the word"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "correct"
        mock_client.chat.completions.create.return_value = mock_response
        
        judge_response_b("ardilla", "squirrel", "response")
        
        call_args = mock_client.chat.completions.create.call_args
        prompt_content = call_args.kwargs['messages'][0]['content']
        assert "ardilla" in prompt_content
