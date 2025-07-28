# Market Impact Analysis: Temporary Price Impact Modeling

## Overview

This repository contains a comprehensive analysis of temporary price impact for market orders based on limit order book (LOB) data. The analysis addresses the quantitative finance problem of modeling the relationship between order size and market impact, providing insights for optimal execution strategies.

## Problem Statement

The analysis addresses the following key questions:

1. **Modeling Temporary Price Impact**: Why is a simple linear model `g_t(x) ≈ β_t * x` a gross oversimplification?
2. **Sophisticated Model Development**: How can we create a more accurate model based on the structure of the limit order book?
3. **Data Analysis**: How can we use real LOB data to validate and fit our models?
4. **Execution Strategy**: How can we formulate strategies to minimize temporary price impact?

## Background

### Market Orders and Limit Orders
- **Market Orders**: Execute immediately at the best available prices, but the price is variable
- **Limit Orders**: Execute at a predetermined price but require waiting in a queue

### Limit Order Book (LOB)
- A snapshot of all active limit orders
- Has a 'bid' side (buyers) and an 'ask' side (sellers)
- Different price levels with corresponding quantities (depth)
- **Mid-Price**: The price exactly between the best bid and the best ask

### Slippage and Temporary Price Impact
- **Slippage**: The difference between the execution price and the mid-price at the time of the order
- For a buy order: Slippage = Execution Price - Mid-Price
- For a sell order: Slippage = Mid-Price - Execution Price
- **Temporary Price Impact g_t(X)**: The slippage incurred when placing a market order for X shares at time t

## Mathematical Framework

### Critique of Linear Model

The simple linear model `g_t(x) ≈ β_t * x` is a gross oversimplification because:

1. **Non-linear Relationship**: The relationship between order size and slippage is non-linear
2. **Piece-wise Structure**: The function appears piece-wise and convex
3. **Order Book Levels**: Large orders consume multiple levels of the order book, resulting in higher average slippage

### Proposed Piece-wise Model

For a buy order of size X, let the ask-side order book at time t be defined by:
- Prices: {p₁, p₂, ..., pₖ} where p₁ < p₂ < ... < pₖ
- Quantities: {q₁, q₂, ..., qₖ}

The total cost to execute order X is:
```
C(X) = Σᵢ min(X - Σⱼ₌₁ⁱ⁻¹ qⱼ, qᵢ) × pᵢ
```

The temporary impact is:
```
g_t(X) = C(X)/X - p_mid
```

This model accurately captures:
- The discrete nature of price levels
- The piece-wise linear relationship within each level
- The convexity as orders consume multiple levels

## Data Structure

The analysis uses LOB data with the following structure:
- **Timestamp**: When the snapshot was taken
- **Symbol**: Stock symbol (CRWV, FROG, SOUN)
- **Price Levels**: Up to 10 levels of bid/ask prices and sizes
- **Format**: CSV files with columns like `bid_px_00`, `ask_px_00`, `bid_sz_00`, `ask_sz_00`, etc.

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python market_impact_analysis.py
```

### Output Files
- `market_impact_analysis_[SYMBOL]_[DATE].png`: Visualization plots
- `market_impact_report.md`: Comprehensive analysis report

## Key Components

### 1. LimitOrderBook Class
- Represents a limit order book snapshot
- Handles bid and ask sides with multiple price levels
- Calculates mid-price and sorts levels appropriately

### 2. TemporaryImpactModel Class
- Implements both linear and piece-wise models
- Fits models to data and makes predictions
- Compares model accuracy

### 3. DataProcessor Class
- Loads and processes LOB data files
- Extracts order book snapshots from raw data
- Handles data validation and cleaning

### 4. MarketImpactAnalyzer Class
- Main analysis engine
- Orchestrates the complete analysis workflow
- Generates visualizations and reports

## Analysis Results

### Model Comparison
The analysis demonstrates:
1. **Linear Model Limitations**: Consistently underestimates impact for large orders
2. **Piece-wise Model Accuracy**: Provides much better fit to actual market structure
3. **Parameter Stability**: Linear beta parameters show significant variation across time

### Key Findings
- Impact functions are consistently convex across all analyzed symbols
- Order book depth significantly affects the shape of the impact function
- Market conditions influence model parameters
- Non-linear models are essential for accurate impact estimation

## Execution Strategy Implications

### Optimal Execution
Given the non-linear nature of market impact, optimal execution strategies should:

1. **Split Large Orders**: Break large orders into smaller pieces
2. **Time Execution**: Consider market conditions and order book depth
3. **Use Limit Orders**: For small orders, limit orders may be preferable
4. **Dynamic Sizing**: Adjust order sizes based on available liquidity

### Mathematical Framework
The optimal execution problem can be formulated as:
```
min Σᵢ g_t(xᵢ) × xᵢ
subject to: Σᵢ xᵢ = X (total order size)
```

## Files in This Repository

- `market_impact_analysis.py`: Main analysis script
- `requirements.txt`: Python dependencies
- `README.md`: This documentation
- `CRWV/`, `FROG/`, `SOUN/`: LOB data directories
- `Blockhouse_Work_Trial_Task-2.pdf`: Original problem statement

## Conclusion

This analysis demonstrates that sophisticated modeling of market impact is essential for effective trading strategies. Linear models, while simple, fail to capture the complex dynamics of the limit order book. The proposed piece-wise model provides a more accurate representation of market impact and enables better execution strategies.

## References

- Original problem statement: Blockhouse_Work_Trial_Task-2.pdf
- Data sources: CRWV, FROG, SOUN LOB data
- Analysis methodology: Piece-wise impact modeling with order book structure 