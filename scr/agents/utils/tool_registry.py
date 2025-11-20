"""
Tool Registry System for Dynamic Tool Loading

This module provides a centralized registry for organizing and managing
tools by category, enabling flexible agent initialization with specific
tool subsets.
"""

from typing import Dict, List, Callable, Optional, Tuple, Any


class ToolRegistry:
    """
    Singleton registry for managing tools organized by category.

    Categories allow agents to selectively load tool subsets:
    - 'search': Web search tools (e.g., Tavily)
    - 'code': Code execution tools (e.g., Python interpreter)
    - 'browser': Web browser navigation tools (e.g., PlayWright)
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern: ensure only one registry instance exists."""
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry with empty category storage."""
        if self._initialized:
            return

        self._categories: Dict[str, Dict[str, Any]] = {}
        # Format: {
        #     'category_name': {
        #         'tools': [...],  # List of OpenAI tool schemas
        #         'functions': {...},  # Dict of tool_name -> callable
        #         'metadata': {...}  # Optional metadata (e.g., browser_manager)
        #     }
        # }
        self._initialized = True

    def register_category(
        self,
        category: str,
        tools: List[Dict],
        tool_functions: Dict[str, Callable],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a tool category with its schemas and implementations.

        Args:
            category: Category name (e.g., 'search', 'code', 'browser')
            tools: List of OpenAI-compatible tool schemas
            tool_functions: Dict mapping tool names to Python callables
            metadata: Optional metadata (e.g., browser_manager for cleanup)

        Raises:
            ValueError: If category already registered
        """
        if category in self._categories:
            raise ValueError(f"Category '{category}' is already registered")

        # Validate that all tools have corresponding functions
        tool_names = {tool['function']['name'] for tool in tools}
        function_names = set(tool_functions.keys())

        if tool_names != function_names:
            missing_funcs = tool_names - function_names
            extra_funcs = function_names - tool_names
            error_parts = []
            if missing_funcs:
                error_parts.append(f"Missing functions: {missing_funcs}")
            if extra_funcs:
                error_parts.append(f"Extra functions: {extra_funcs}")
            raise ValueError(
                f"Tool/function mismatch in category '{category}': {', '.join(error_parts)}"
            )

        self._categories[category] = {
            'tools': tools,
            'functions': tool_functions,
            'metadata': metadata or {}
        }

    def get_tools_by_categories(
        self,
        categories: Optional[List[str]] = None
    ) -> Tuple[List[Dict], Dict[str, Callable]]:
        """
        Get tools and functions for specified categories.

        Args:
            categories: List of category names to load, or None for all categories

        Returns:
            Tuple of (tool_schemas, tool_functions):
                - tool_schemas: List of OpenAI-compatible tool schemas
                - tool_functions: Dict mapping tool names to callables

        Raises:
            ValueError: If a requested category doesn't exist
        """
        # If None, return all categories
        if categories is None:
            categories = list(self._categories.keys())

        # Validate all categories exist
        for category in categories:
            if category not in self._categories:
                available = list(self._categories.keys())
                raise ValueError(
                    f"Unknown category '{category}'. Available: {available}"
                )

        # Merge tools and functions from selected categories
        merged_tools = []
        merged_functions = {}

        for category in categories:
            cat_data = self._categories[category]
            merged_tools.extend(cat_data['tools'])
            merged_functions.update(cat_data['functions'])

        return merged_tools, merged_functions

    def get_category_metadata(self, category: str) -> Dict[str, Any]:
        """
        Get metadata for a specific category.

        Args:
            category: Category name

        Returns:
            Metadata dictionary for the category

        Raises:
            ValueError: If category doesn't exist
        """
        if category not in self._categories:
            raise ValueError(f"Category '{category}' not found")
        return self._categories[category]['metadata']

    def get_all_metadata(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get merged metadata from multiple categories.

        Args:
            categories: List of categories, or None for all

        Returns:
            Merged metadata dict from all requested categories
        """
        if categories is None:
            categories = list(self._categories.keys())

        merged_metadata = {}
        for category in categories:
            if category in self._categories:
                merged_metadata.update(self._categories[category]['metadata'])

        return merged_metadata

    def list_categories(self) -> List[str]:
        """
        List all registered category names.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def category_exists(self, category: str) -> bool:
        """
        Check if a category is registered.

        Args:
            category: Category name to check

        Returns:
            True if category exists, False otherwise
        """
        return category in self._categories

    def clear(self) -> None:
        """
        Clear all registered categories.

        Warning: This is primarily for testing. Use with caution.
        """
        self._categories.clear()


# Global singleton instance -> returns all tools
tool_registry = ToolRegistry()
