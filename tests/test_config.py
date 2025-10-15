import importlib

import pytest


config = importlib.import_module("ui_screenshot_to_prompt.config")


def test_set_detection_method_updates_term():
    original_method = config.get_detection_method()
    try:
        config.set_detection_method("advanced")
        assert config.get_detection_method() == "advanced"
        assert config.get_detection_term() == "component"

        config.set_detection_method("basic")
        assert config.get_detection_method() == "basic"
        assert config.get_detection_term() == "region"
    finally:
        if original_method and original_method != config.get_detection_method():
            config.set_detection_method(original_method)


def test_set_detection_method_invalid():
    with pytest.raises(ValueError):
        config.set_detection_method("unknown")


def test_set_prompt_choice_invalid():
    with pytest.raises(ValueError):
        config.set_prompt_choice("verbose")


def test_build_super_prompt_respects_prompt_choice():
    original_prompt = config.get_prompt_choice()
    try:
        config.set_prompt_choice("concise")
        prompt = config.build_super_prompt(
            main_image_caption="Layout overview",
            region_descriptions=["Header", "Content"],
            activity_description="Browsing",
        )
        assert "This study presents" in prompt

        config.set_prompt_choice("extensive")
        prompt = config.build_super_prompt(
            main_image_caption="Layout overview",
            region_descriptions=["Header", "Content"],
            activity_description="Browsing",
        )
        assert "You are an expert UI development agent" in prompt
    finally:
        config.set_prompt_choice(original_prompt)
