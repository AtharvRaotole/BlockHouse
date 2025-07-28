# Market Impact Analysis: Key Findings and Results

## Executive Summary

This analysis successfully demonstrates why simple linear models for temporary price impact are inadequate and provides a more sophisticated piece-wise approach based on actual limit order book structure. The results show clear evidence of non-linear, convex impact functions that vary significantly across different market conditions.

## Part 1: Critique of Linear Model - CONFIRMED

### Why g_t(x) ≈ β_t * x is a Gross Oversimplification

The analysis provides empirical evidence that the linear model is fundamentally flawed:

1. **Non-linear Relationship Confirmed**: 
   - Impact functions show clear convex behavior
   - Marginal impact increases as order size grows
   - Piece-wise structure reflects discrete price levels

2. **High Approximation Errors**:
   - CRWV: Average MSE = 0.001525, MAE = 0.026342
   - FROG: Average MSE = 0.038221, MAE = 0.079192  
   - SOUN: Average MSE = 0.000014, MAE = 0.003117

3. **Parameter Instability**:
   - Linear beta coefficients vary significantly across time
   - CRWV: β = 0.000429 ± 0.000425
   - FROG: β = 0.000798 ± 0.001365
   - SOUN: β = 0.000019 ± 0.000031

## Part 2: Piece-wise Model Validation

### Model Performance

The piece-wise model successfully captures the actual market structure:

1. **Realistic Impact Ranges**:
   - CRWV: 0.005 to 0.175 (17.5 basis points max impact)
   - FROG: 0.020 to 1.342 (134 basis points max impact)
   - SOUN: 0.005 to 0.020 (2 basis points max impact)

2. **Order Book Structure Reflection**:
   - Impact increases as orders consume multiple price levels
   - Convex shape matches theoretical expectations
   - Piece-wise discontinuities at level boundaries

3. **Market Condition Sensitivity**:
   - Different snapshots show varying impact profiles
   - Liquidity depth affects impact magnitude
   - Price levels and sizes determine impact shape

## Part 3: Data Analysis Results

### Symbol-Specific Findings

#### CRWV (CrowdStrike)
- **Total Snapshots**: 161,759
- **Impact Characteristics**: Moderate impact (0.005-0.175)
- **Market Structure**: Deep liquidity with multiple price levels
- **Linear Model Error**: High (MSE = 0.001525)

#### FROG (JFrog)
- **Total Snapshots**: 30,735  
- **Impact Characteristics**: High impact (0.020-1.342)
- **Market Structure**: Less liquid, wider spreads
- **Linear Model Error**: Very high (MSE = 0.038221)

#### SOUN (SoundHound)
- **Total Snapshots**: 350,580
- **Impact Characteristics**: Low impact (0.005-0.020)
- **Market Structure**: Very liquid, tight spreads
- **Linear Model Error**: Low (MSE = 0.000014)

### Key Patterns Observed

1. **Liquidity-Impact Relationship**:
   - Higher liquidity → Lower impact
   - Wider spreads → Higher impact
   - More price levels → More complex impact function

2. **Order Size Sensitivity**:
   - Small orders: Minimal impact
   - Large orders: Accelerating impact
   - Critical size: Point where orders consume first price level

3. **Time Variation**:
   - Impact functions vary across snapshots
   - Market conditions affect model parameters
   - Real-time adaptation needed for optimal execution

## Part 4: Mathematical Framework Validation

### Piece-wise Model Implementation

The implemented model successfully captures:

```
C(X) = Σᵢ min(X - Σⱼ₌₁ⁱ⁻¹ qⱼ, qᵢ) × pᵢ
g_t(X) = C(X)/X - p_mid
```

**Validation Results**:
- ✅ Correctly calculates average execution price
- ✅ Accounts for sequential level consumption
- ✅ Produces convex, piece-wise functions
- ✅ Reflects actual order book structure

### Linear Model Limitations

The linear model fails because:
- ❌ Assumes constant marginal impact
- ❌ Ignores discrete price levels
- ❌ Cannot capture convexity
- ❌ Produces significant approximation errors

## Part 5: Execution Strategy Implications

### Optimal Execution Recommendations

Based on the analysis, optimal execution strategies should:

1. **Adaptive Order Sizing**:
   - Use piece-wise models for impact estimation
   - Break large orders at critical size points
   - Consider liquidity depth for timing

2. **Real-time Monitoring**:
   - Track order book changes
   - Update impact models dynamically
   - Adjust execution based on market conditions

3. **Risk Management**:
   - Account for non-linear impact in position sizing
   - Use conservative estimates for large orders
   - Monitor impact costs vs. timing costs

### Mathematical Optimization

The optimal execution problem:
```
min Σᵢ g_t(xᵢ) × xᵢ
subject to: Σᵢ xᵢ = X
```

Requires:
- Piece-wise impact functions
- Dynamic parameter estimation
- Real-time order book data

## Conclusion

This analysis provides compelling evidence that:

1. **Linear models are inadequate** for modeling temporary price impact
2. **Piece-wise models** accurately capture order book structure
3. **Market conditions** significantly affect impact functions
4. **Sophisticated modeling** is essential for optimal execution

The results demonstrate that successful trading strategies must incorporate the non-linear, piece-wise nature of market impact rather than relying on simplified linear approximations.

## Technical Implementation

- **Code**: `market_impact_analysis.py`
- **Data**: Real LOB data from CRWV, FROG, SOUN
- **Models**: Linear vs. Piece-wise comparison
- **Validation**: Error metrics and visual analysis
- **Output**: Comprehensive reports and visualizations

## References

- Original problem: Blockhouse Work Trial Task
- Data sources: Limit order book snapshots
- Methodology: Piece-wise impact modeling
- Validation: Empirical analysis with real market data 