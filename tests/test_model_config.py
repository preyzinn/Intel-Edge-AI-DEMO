import pytest

from edge_ai_demo.model.model_config import (
    DEFAULT_MODEL_ID,
    SUPPORTED_MODELS,
    GenerationSettings,
    ModelConfig,
    get_model_config,
)


def test_supported_model_allowlist_resolves_default_model() -> None:
    model = get_model_config(DEFAULT_MODEL_ID)

    assert model in SUPPORTED_MODELS
    assert model.model_id == DEFAULT_MODEL_ID
    assert model.display_name
    assert model.revision


def test_model_outside_allowlist_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model_config("arbitrary/unsupported-model")


def test_model_cache_key_is_stable_and_filesystem_safe() -> None:
    model = ModelConfig(
        model_id="Example Org/Small_Model@v1",
        display_name="Small Model",
        revision="1234567890abcdef",
    )

    assert model.cache_key == "example-org-small-model-v1-1234567890ab"
    assert model.cache_key == model.cache_key


def test_greedy_generation_kwargs_omit_sampling_only_values() -> None:
    settings = GenerationSettings(
        prompt="Hello",
        max_new_tokens=17,
        temperature=0.0,
        top_p=0.4,
        repetition_penalty=1.2,
    )

    assert settings.generate_kwargs() == {
        "max_new_tokens": 17,
        "repetition_penalty": 1.2,
        "do_sample": False,
        "use_cache": True,
    }


def test_sampling_generation_kwargs_include_temperature_and_top_p() -> None:
    settings = GenerationSettings(
        prompt="Hello",
        max_new_tokens=23,
        temperature=0.7,
        top_p=0.85,
        repetition_penalty=1.05,
        seed=7,
    )

    assert settings.generate_kwargs() == {
        "max_new_tokens": 23,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "use_cache": True,
        "temperature": 0.7,
        "top_p": 0.85,
    }
