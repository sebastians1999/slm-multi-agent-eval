"""Quick test script for cost tracking functionality."""

from scr.utilities.cost_tracker import CostTracker


def test_cost_tracker():
    """Test the cost tracker with sample data."""
    # Initialize tracker
    tracker = CostTracker()
    # Test with a common model (adjust model name as needed)
    model_name = "Magistral Medium 1.2"
    input_tokens = 1000
    output_tokens = 500

    print(f"Testing cost calculation for: {model_name}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}\n")

    # Calculate cost
    cost_data = tracker.calculate_cost(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    if cost_data:
        print("✓ Cost calculation successful!")
        print(f"\nCost breakdown:")
        print(f"  Input cost:  ${cost_data['input_cost']:.6f}")
        print(f"  Output cost: ${cost_data['output_cost']:.6f}")
        print(f"  Total cost:  ${cost_data['total_cost']:.6f}")
        print(f"\nPricing per 1M tokens:")
        print(f"  Input:  ${cost_data['price_per_1m_input_tokens']}")
        print(f"  Output: ${cost_data['price_per_1m_output_tokens']}")
    else:
        print("✗ Cost calculation failed!")
        print("Check that ARTIFICIAL_ANALYSIS_API_KEY is set in .env")


if __name__ == "__main__":
    test_cost_tracker()
