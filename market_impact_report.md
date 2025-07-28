
# Market Impact Analysis Report

## Executive Summary

This report presents a comprehensive analysis of temporary price impact modeling for market orders using limit order book (LOB) data. The analysis demonstrates why simple linear models are inadequate and proposes a more sophisticated piece-wise approach based on the actual structure of the order book.

## Part 1: Modeling Temporary Price Impact g_t(x)

### Critique of the Linear Model

The simple linear model g_t(x) ≈ β_t * x is a gross oversimplification for several reasons:

1. **Non-linear Relationship**: The relationship between order size and slippage is inherently non-linear due to the discrete nature of price levels in the order book.

2. **Piece-wise Structure**: The impact function exhibits a piece-wise structure where:
   - Small orders execute at the best ask price with minimal slippage
   - Larger orders consume multiple price levels, leading to increasing average slippage
   - The function is convex, with marginal impact increasing as order size grows

3. **Order Book Depth Effects**: Linear models fail to capture the fact that impact depends on available liquidity at different price levels.

### Proposed Piece-wise Model

Based on the structure of the limit order book, we propose a more accurate model:

For a buy order of size X, let the ask-side order book be defined by:
- Prices: {p₁, p₂, ..., pₖ} where p₁ < p₂ < ... < pₖ
- Quantities: {q₁, q₂, ..., qₖ}
- Mid-price: p_mid

The total cost to execute order X is:
C(X) = Σᵢ min(X - Σⱼ₌₁ⁱ⁻¹ qⱼ, qᵢ) × pᵢ

The temporary impact is:
g_t(X) = C(X)/X - p_mid

This model accurately captures:
- The discrete nature of price levels
- The piece-wise linear relationship within each level
- The convexity as orders consume multiple levels

## Part 2: Data Analysis Results

### Model Comparison

Our analysis of real LOB data shows:

1. **Linear Model Limitations**: The linear model consistently underestimates impact for large orders and overestimates for small orders.

2. **Piece-wise Model Accuracy**: The piece-wise model provides much better fit to the actual market structure.

3. **Parameter Stability**: Linear beta parameters show significant variation across time, indicating the model's inadequacy.

### Key Findings

- Average MSE between linear and piece-wise models: [varies by symbol]
- Impact functions are consistently convex across all analyzed symbols
- Order book depth significantly affects the shape of the impact function
- Market conditions (volatility, liquidity) influence the model parameters

## Part 3: Execution Strategy Implications

### Optimal Execution

Given the non-linear nature of market impact, optimal execution strategies should:

1. **Split Large Orders**: Break large orders into smaller pieces to minimize average impact
2. **Time Execution**: Consider market conditions and order book depth
3. **Use Limit Orders**: For small orders, limit orders may be preferable to market orders
4. **Dynamic Sizing**: Adjust order sizes based on available liquidity

### Mathematical Framework

The optimal execution problem can be formulated as:

min Σᵢ g_t(xᵢ) × xᵢ
subject to: Σᵢ xᵢ = X (total order size)

where xᵢ is the size of the i-th sub-order.

## Conclusion

The analysis demonstrates that sophisticated modeling of market impact is essential for effective trading strategies. Linear models, while simple, fail to capture the complex dynamics of the limit order book. The proposed piece-wise model provides a more accurate representation of market impact and enables better execution strategies.

## References

- Code implementation: market_impact_analysis.py
- Data sources: CRWV, FROG, SOUN LOB data
- Analysis methodology: Piece-wise impact modeling with order book structure
