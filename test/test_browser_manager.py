"""
Tests for Browser Lifecycle Manager

Tests browser initialization, reuse, and cleanup functionality.
Note: These tests require playwright to be installed.
"""

import pytest
from scr.agents.utils.browser_manager import BrowserManager


@pytest.fixture
def browser_manager():
    """Fixture that provides a browser manager and cleans up after test."""
    manager = BrowserManager()
    yield manager
    manager.close()


def test_browser_lazy_initialization(browser_manager):
    """Test that browser is not created until first get_browser() call."""
    # Browser should not be created yet
    assert browser_manager._browser is None
    assert browser_manager._playwright is None

    # First call creates browser
    browser = browser_manager.get_browser()
    assert browser is not None
    assert browser_manager._browser is not None
    assert browser_manager._playwright is not None


def test_browser_reuse(browser_manager):
    """Test that same browser instance is returned on subsequent calls."""
    browser1 = browser_manager.get_browser()
    browser2 = browser_manager.get_browser()

    assert browser1 is browser2


def test_browser_close(browser_manager):
    """Test that close() properly cleans up resources."""
    # Initialize browser
    browser_manager.get_browser()

    # Close should cleanup
    browser_manager.close()
    assert browser_manager._browser is None
    assert browser_manager._playwright is None
    assert browser_manager.is_closed()


def test_browser_close_multiple_times():
    """Test that close() can be called multiple times safely."""
    manager = BrowserManager()
    manager.get_browser()

    # Multiple closes should not raise errors
    manager.close()
    manager.close()
    manager.close()


def test_browser_usage_after_close(browser_manager):
    """Test that using browser after close raises error."""
    browser_manager.get_browser()
    browser_manager.close()

    with pytest.raises(RuntimeError, match="has been closed"):
        browser_manager.get_browser()


def test_browser_context_manager():
    """Test using browser manager as context manager."""
    with BrowserManager() as manager:
        browser = manager.get_browser()
        assert browser is not None
        assert not manager.is_closed()

    # Should be closed after exiting context
    assert manager.is_closed()


def test_browser_crash_recovery(browser_manager):
    """Test that browser can recover from crashes."""
    browser1 = browser_manager.get_browser()

    # Simulate crash by forcefully closing browser
    try:
        browser1.close()
    except:
        pass

    # Get browser again should create new instance
    browser2 = browser_manager.get_browser()
    assert browser2 is not None
    # Should be different instance after recovery
    assert browser2 is not browser1


@pytest.mark.skipif(
    True,  # Skip by default as it requires playwright installation
    reason="Requires playwright browser binaries (run: playwright install chromium)"
)
def test_browser_actual_navigation():
    """Integration test: actually navigate to a page."""
    with BrowserManager() as manager:
        browser = manager.get_browser()
        page = browser.new_page()

        # Navigate to a simple page
        page.goto("https://www.example.com")
        assert "Example Domain" in page.title()

        page.close()
