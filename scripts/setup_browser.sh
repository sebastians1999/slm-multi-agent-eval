#!/bin/bash
#
# Browser Setup Script
#
# This script installs Playwright browser binaries needed for browser tools.
# Run this after installing Python dependencies with uv.
#
# Usage:
#   bash scripts/setup_browser.sh
#   # OR make executable first:
#   # chmod +x scripts/setup_browser.sh && ./scripts/setup_browser.sh
#

set -e  # Exit on error

echo "Setting up Playwright browser for browser tools..."
echo ""

# Check if playwright is installed
if ! python -c "import playwright" 2>/dev/null; then
    echo "Error: playwright package not installed."
    echo "Please install dependencies first with uv:"
    echo "  uv sync"
    exit 1
fi

echo "Installing Chromium browser binary..."
echo "(This may take a few minutes...)"
playwright install chromium

echo ""
echo "✓ Browser setup complete!"
echo ""
echo "You can now use browser tools in your agents:"
echo "  from scr.agents.base_agent import BaseAgent"
echo "  agent = BaseAgent(model='gpt-4', tool_categories=['browser'])"
echo ""
echo "Or load all tools (including browser):"
echo "  agent = BaseAgent(model='gpt-4')  # All tools loaded by default"
