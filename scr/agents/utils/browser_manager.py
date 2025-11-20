"""
Browser Lifecycle Manager

Manages shared Playwright browser instances for browser-based tools.
Implements lazy initialization and proper resource cleanup to ensure
efficient browser reuse across multiple tool calls.
"""

from typing import Optional, Any
import atexit


class BrowserManager:
    """
    Manages lifecycle of a shared Playwright browser instance.

    Features:
    - Lazy initialization: browser created only when first needed
    - Shared instance: same browser reused across all tool calls
    - Automatic cleanup: ensures browser closed on exit
    - Context manager support: use with 'with' statement
    - Error recovery: handles browser crashes gracefully

    Example:
        >>> manager = BrowserManager()
        >>> browser = manager.get_browser()  # Creates browser on first call
        >>> # Use browser...
        >>> browser2 = manager.get_browser()  # Returns same instance
        >>> manager.close()  # Cleanup when done

    Or with context manager:
        >>> with BrowserManager() as manager:
        ...     browser = manager.get_browser()
        ...     # Use browser...
        ... # Automatically cleaned up
    """

    def __init__(self):
        """Initialize browser manager with lazy loading."""
        self._playwright: Optional[Any] = None
        self._browser: Optional[Any] = None
        self._is_closed: bool = False

        # Register cleanup on program exit
        atexit.register(self.close)

    def get_browser(self) -> Any:
        """
        Get the browser instance, creating it if necessary.

        Returns:
            Playwright browser instance (sync)

        Raises:
            RuntimeError: If browser creation fails or manager is closed

        Note:
            First call initializes Playwright and creates browser.
            Subsequent calls return the cached instance.
        """
        if self._is_closed:
            raise RuntimeError("BrowserManager has been closed")

        # Lazy initialization
        if self._browser is None:
            self._initialize_browser()

        # Verify browser is still alive
        if not self._is_browser_alive():
            print("[BrowserManager] Browser crashed, recreating...")
            self._cleanup_browser()
            self._initialize_browser()

        return self._browser

    def _initialize_browser(self) -> None:
        """
        Initialize Playwright and create browser instance.

        Raises:
            ImportError: If playwright not installed
            RuntimeError: If browser creation fails
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "Playwright not installed. Install with: pip install playwright && "
                "playwright install chromium"
            ) from e

        try:
            # Start Playwright
            self._playwright = sync_playwright().start()

            # Launch browser (chromium is most compatible)
            self._browser = self._playwright.chromium.launch(
                headless=True,  # Run in headless mode for efficiency
                # Optional: add args for better compatibility
                # args=['--disable-blink-features=AutomationControlled']
            )

            print("[BrowserManager] Browser initialized successfully")

        except Exception as e:
            # Cleanup on failure
            self._cleanup_browser()
            raise RuntimeError(f"Failed to initialize browser: {e}") from e

    def _is_browser_alive(self) -> bool:
        """
        Check if browser is still running.

        Returns:
            True if browser is alive, False otherwise
        """
        if self._browser is None:
            return False

        try:
            # Check if browser is still connected
            # is_connected() returns False if browser was closed
            return self._browser.is_connected()
        except Exception:
            return False

    def _cleanup_browser(self) -> None:
        """
        Cleanup browser and playwright resources.

        Safe to call multiple times.
        """
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as e:
                print(f"[BrowserManager] Error closing browser: {e}")
            finally:
                self._browser = None

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception as e:
                print(f"[BrowserManager] Error stopping playwright: {e}")
            finally:
                self._playwright = None

    def close(self) -> None:
        """
        Close browser and cleanup all resources.

        Safe to call multiple times. After calling close(), the manager
        cannot be reused.
        """
        if self._is_closed:
            return

        self._cleanup_browser()
        self._is_closed = True
        print("[BrowserManager] Closed successfully")

    def is_closed(self) -> bool:
        """
        Check if manager has been closed.

        Returns:
            True if closed, False otherwise
        """
        return self._is_closed

    # Context manager protocol
    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
