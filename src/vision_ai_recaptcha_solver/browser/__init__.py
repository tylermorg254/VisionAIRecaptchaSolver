"""Browser navigation module."""

from vision_ai_recaptcha_solver.browser.navigation import (
    click_checkbox,
    click_reload_button,
    click_tile,
    click_verify_button,
    get_captcha_image_urls,
    get_challenge_iframe,
    get_challenge_title,
    get_checkbox_iframe,
    get_target_keyword,
    is_solved,
    is_verify_button_disabled,
    wait_for_verify_result,
)

__all__ = [
    "get_checkbox_iframe",
    "get_challenge_iframe",
    "click_checkbox",
    "click_verify_button",
    "click_reload_button",
    "click_tile",
    "is_solved",
    "is_verify_button_disabled",
    "wait_for_verify_result",
    "get_target_keyword",
    "get_challenge_title",
    "get_captcha_image_urls",
]
