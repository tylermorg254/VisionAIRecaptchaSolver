"""Synchronous RecaptchaSolver implementation."""

from __future__ import annotations

import atexit
import contextlib
import shutil
import signal
import sys
import time
import weakref
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from recaptcha_domain_replicator import RecaptchaDomainReplicator

from vision_ai_recaptcha_solver.browser.navigation import (
    click_checkbox,
    click_reload_button,
    click_verify_button,
    get_challenge_iframe,
    get_challenge_title,
    get_target_keyword,
    is_solved,
    wait_for_verify_result,
)
from vision_ai_recaptcha_solver.captcha.dynamic_handler import DynamicCaptchaHandler
from vision_ai_recaptcha_solver.captcha.selection_handler import SelectionCaptchaHandler
from vision_ai_recaptcha_solver.captcha.square_handler import SquareCaptchaHandler
from vision_ai_recaptcha_solver.config import SolverConfig
from vision_ai_recaptcha_solver.detector.yolo_detector import YOLODetector
from vision_ai_recaptcha_solver.exceptions import (
    CaptchaNotFoundError,
    ElementNotFoundError,
    LowConfidenceError,
    RecaptchaSolverError,
    TokenExtractionError,
    UnsupportedCaptchaError,
)
from vision_ai_recaptcha_solver.logging_config import setup_logging
from vision_ai_recaptcha_solver.resource_allocation import (
    release_solver_resources,
    reserve_solver_resources,
)
from vision_ai_recaptcha_solver.types import CaptchaType, SolveResult
from vision_ai_recaptcha_solver.utils import human_delay

if TYPE_CHECKING:
    from vision_ai_recaptcha_solver.captcha.base_handler import BaseCaptchaHandler


# Global registry for active solver instances
_active_solvers: weakref.WeakSet[RecaptchaSolver] = weakref.WeakSet()
_original_sigint_handler: Any = None
_original_sigterm_handler: Any = None
_cleanup_registered: bool = False
_DOWNLOAD_DIR_MARKER = ".vision_ai_recaptcha_solver_owned"


def _cleanup_all_solvers() -> None:
    """Cleanup all active solver instances."""
    for solver in list(_active_solvers):
        with contextlib.suppress(Exception):
            solver.close()


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals by cleaning up browsers before exit."""
    _cleanup_all_solvers()
    if signum == signal.SIGINT and _original_sigint_handler:
        if callable(_original_sigint_handler):
            _original_sigint_handler(signum, frame)
        else:
            sys.exit(130)
    elif signum == signal.SIGTERM and _original_sigterm_handler:
        if callable(_original_sigterm_handler):
            _original_sigterm_handler(signum, frame)
        else:
            sys.exit(143)
    else:
        sys.exit(128 + signum)


def _register_cleanup_handlers(register_signal_handlers: bool = True) -> None:
    global _cleanup_registered, _original_sigint_handler, _original_sigterm_handler
    if _cleanup_registered:
        return
    _cleanup_registered = True

    atexit.register(_cleanup_all_solvers)

    if not register_signal_handlers:
        return

    _original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _signal_handler)

    if hasattr(signal, "SIGTERM"):
        _original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _signal_handler)


class RecaptchaSolver:
    """Synchronous reCAPTCHA solver using YOLO object detection."""

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()
        self.logger = setup_logging(self.config.log_level, "vision_ai_recaptcha_solver")
        reserve_solver_resources(self, self.config, self.logger)
        self._owns_download_dir: bool = False
        self._init_download_dir()

        # NEW: Store browser reference
        self.browser: Any | None = None

        self._detector = YOLODetector(
            model_path=self.config.model_path,
            detection_model_path=self.config.detection_model_path,
            verbose=self.config.verbose,
            logger=self.logger,
            conf_threshold=self.config.conf_threshold,
            fourth_cell_threshold=self.config.fourth_cell_threshold,
            detection_conf_threshold=self.config.detection_conf_threshold,
        )

        self._handlers: dict[CaptchaType, BaseCaptchaHandler] = {
            CaptchaType.DYNAMIC_3X3: DynamicCaptchaHandler(
                self._detector, self.config, self.logger
            ),
            CaptchaType.SELECTION_3X3: SelectionCaptchaHandler(
                self._detector, self.config, self.logger
            ),
            CaptchaType.SQUARE_4X4: SquareCaptchaHandler(self._detector, self.config, self.logger),
        }

        self._replicator: RecaptchaDomainReplicator | None = None
        self._closed: bool = False

        _active_solvers.add(self)
        _register_cleanup_handlers(register_signal_handlers=self.config.register_signal_handlers)

    # ... ( _init_download_dir, __enter__, __exit__ remain unchanged ) ...

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        _active_solvers.discard(self)

        self._cleanup_replicator()
        if self.config.cleanup_tmp_on_close:
            self._cleanup_tmp_directory()

        release_solver_resources(self)

    def _cleanup_replicator(self) -> None:
        if self._replicator:
            # MODIFIED: Conditional browser closing
            keep_open = getattr(self.config, 'keep_browser_open', False)
            if keep_open and self.browser:
                self.logger.info("Browser kept open intentionally (keep_browser_open=True)")
            else:
                try:
                    self._replicator.close_browser()
                    self.logger.debug("Browser closed")
                except Exception as e:
                    self.logger.debug(f"Error closing browser: {e}")

            try:
                self._replicator.stop_http_server()
            except Exception as e:
                self.logger.debug(f"Error stopping server: {e}")
            finally:
                self._replicator = None

    def _cleanup_tmp_directory(self) -> None:
        # unchanged
        if not self._owns_download_dir:
            self.logger.debug("Skipping download directory cleanup: %s", self.config.download_dir)
            return
        try:
            if self.config.download_dir.exists():
                shutil.rmtree(self.config.download_dir)
                self.logger.debug(f"Cleaned up temporary directory: {self.config.download_dir}")
        except Exception as e:
            self.logger.debug(f"Error cleaning up temporary directory: {e}")

    # NEW: Manual close method when keep_browser_open is used
    def force_close_browser(self) -> None:
        """Force close the kept-open browser when you're done with it."""
        if self.browser and self._replicator:
            try:
                self._replicator.close_browser()
                self.logger.info("Browser forcefully closed")
            except Exception as e:
                self.logger.debug(f"Error in force_close_browser: {e}")
            self.browser = None

    def solve(
        self,
        website_key: str,
        website_url: str,
        *,
        is_invisible: bool = False,
        action: str | None = None,
        is_enterprise: bool = False,
        api_domain: str = "google.com",
        bypass_domain_check: bool = True,
        use_ssl: bool = True,
        cookies: list[dict[str, Any]] | None = None,
        user_agent: str | None = None,
    ) -> SolveResult:
        if not website_key or not website_key.strip():
            raise ValueError("website_key cannot be empty")
        if not website_url or not website_url.strip():
            raise ValueError("website_url cannot be empty")

        parsed = urlparse(website_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {website_url}")

        if self._closed:
            raise RecaptchaSolverError("Solver has been closed")

        start_time = time.time()
        attempts = 0
        last_captcha_type = CaptchaType.UNKNOWN

        try:
            self._cleanup_replicator()

            self._replicator = RecaptchaDomainReplicator(
                download_dir=str(self.config.download_dir),
                server_port=self.config.server_port,
                persist_html=self.config.persist_html,
                proxy=self.config.proxy,
                browser_path=self.config.browser_path,
            )

            cookies_payload: Any = cookies
            browser, token_handle = self._replicator.replicate_captcha(
                website_key=website_key,
                website_url=website_url,
                is_invisible=is_invisible,
                action=action,
                is_enterprise=is_enterprise,
                api_domain=api_domain,
                bypass_domain_check=bypass_domain_check,
                use_ssl=use_ssl,
                cookies=cookies_payload,
                user_agent=user_agent,
                headless=self.config.headless,
            )

            if not browser:
                raise CaptchaNotFoundError("Failed to initialize browser session")

            # NEW: Store the browser reference
            self.browser = browser

            # Invisible v3 path
            if is_invisible:
                self.logger.info("Invisible reCAPTCHA (v3) detected - waiting for token...")
                last_captcha_type = CaptchaType.INVISIBLE

                token = token_handle.wait(timeout=self.config.timeout) if token_handle else None
                if not token:
                    raise TokenExtractionError("Failed to extract reCAPTCHA v3 token")

                result_cookies = self._get_cookies(browser)
                time_taken = round(time.time() - start_time, 2)

                self.logger.info("reCAPTCHA token obtained successfully!")
                return SolveResult(
                    token=token,
                    cookies=result_cookies,
                    time_taken=time_taken,
                    captcha_type=CaptchaType.INVISIBLE,
                    attempts=0,
                    browser=self.browser   # ← returned here
                )

            # v2 visible path
            human_delay(mean=0.8, sigma=0.2)
            try:
                click_checkbox(browser)
            except ElementNotFoundError as e:
                raise CaptchaNotFoundError(f"Could not find captcha checkbox: {e}") from e

            human_delay(mean=0.5, sigma=0.3)

            if is_solved(browser, timeout=2):
                self.logger.info("Captcha solved immediately on checkbox click (no challenge)")
                token = token_handle.wait(timeout=self.config.timeout) if token_handle else None
                if not token:
                    raise TokenExtractionError("Failed to extract reCAPTCHA token after immediate solve")

                result_cookies = self._get_cookies(browser)
                time_taken = round(time.time() - start_time, 2)

                return SolveResult(
                    token=token,
                    cookies=result_cookies,
                    time_taken=time_taken,
                    captcha_type=CaptchaType.NO_CHALLENGE,
                    attempts=0,
                    browser=self.browser   # ← returned here
                )

            self._detector.ensure_warmup_complete()

            while attempts < self.config.max_attempts:
                attempts += 1
                self.logger.debug(f"Solve attempt {attempts}/{self.config.max_attempts}")

                try:
                    captcha_type = self._determine_captcha_type(browser)
                    last_captcha_type = captcha_type
                    target_class = self._get_target_class(browser)

                    if target_class is None:
                        self.logger.info("Unknown target, reloading captcha")
                        click_reload_button(browser)
                        human_delay(mean=self.config.human_delay_mean, sigma=self.config.human_delay_sigma)
                        challenge_frame = get_challenge_iframe(browser, timeout=self.config.default_timeout)
                        if challenge_frame:
                            challenge_frame.ele("#rc-imageselect-target td", timeout=self.config.default_timeout)
                        continue

                    handler = self._get_handler(captcha_type)
                    clicked_cells = handler.solve(browser, target_class)

                    if not clicked_cells:
                        self.logger.info("No cells clicked, reloading")
                        click_reload_button(browser)
                        human_delay(mean=self.config.human_delay_mean, sigma=self.config.human_delay_sigma)
                        challenge_frame = get_challenge_iframe(browser, timeout=self.config.default_timeout)
                        if challenge_frame:
                            challenge_frame.ele("#rc-imageselect-target td", timeout=self.config.default_timeout)
                        continue

                    human_delay(mean=0.3, sigma=0.2)
                    click_verify_button(browser)

                    if wait_for_verify_result(browser, timeout=self.config.default_timeout):
                        self.logger.info("Captcha solved successfully!")
                        break

                    human_delay(mean=0.2, sigma=0.1)

                except LowConfidenceError as e:
                    self.logger.info(f"Low confidence detection, reloading: {e}")
                    click_reload_button(browser)
                    human_delay(mean=self.config.human_delay_mean, sigma=self.config.human_delay_sigma)
                    challenge_frame = get_challenge_iframe(browser, timeout=self.config.default_timeout)
                    if challenge_frame:
                        challenge_frame.ele("#rc-imageselect-target td", timeout=self.config.default_timeout)

                except (ElementNotFoundError, UnsupportedCaptchaError) as e:
                    self.logger.warning(f"Attempt {attempts} failed: {e}")
                    human_delay(mean=0.5, sigma=0.1)

            token = token_handle.wait(timeout=self.config.timeout) if token_handle else None
            if not token:
                raise TokenExtractionError("Failed to extract reCAPTCHA token")

            result_cookies = self._get_cookies(browser)
            time_taken = round(time.time() - start_time, 2)

            return SolveResult(
                token=token,
                cookies=result_cookies,
                time_taken=time_taken,
                captcha_type=last_captcha_type,
                attempts=attempts,
                browser=self.browser   # ← returned here
            )

        except RecaptchaSolverError:
            raise
        except (RuntimeError, OSError, ValueError) as e:
            raise RecaptchaSolverError(f"Solve failed: {e}") from e

    # ... ( _determine_captcha_type, _get_target_class, _get_handler, _get_cookies remain unchanged ) ...
