"""Shared pytest fixtures and configuration for all tests."""

import pytest


@pytest.fixture
def sample_models():
    """Standard set of model names for testing"""
    return ["gemma3:12b", "llama3.1:latest"]


@pytest.fixture
def sample_prompts():
    """Standard prompts for testing"""
    return {
        "prompt_a": "Define la palabra {word} en español",
        "prompt_b": "Escribe dos frases sobre {word}"
    }


@pytest.fixture
def sample_vocabulary():
    """Standard vocabulary entries for testing"""
    return [
        {"word": "ardilla", "answer": "Un roedor pequeño que vive en árboles"},
        {"word": "corbata", "answer": "Una prenda de vestir que se lleva alrededor del cuello"},
        {"word": "agüista", "answer": "Persona que toma aguas medicinales"}
    ]
