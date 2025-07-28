#!/usr/bin/env python3
"""
Execution Strategy: Optimal Order Splitting to Minimize Temporary Price Impact

This module implements Part 2 of the Blockhouse Work Trial Task:
Formulating an execution strategy to minimize temporary price impact when
executing a large order S over the trading day.

Author: Quantitative Finance Analysis
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ExecutionParameters:
    """Parameters for execution strategy optimization."""
    total_shares: int
    trading_hours: float = 6.5  # Market hours (9:30 AM - 4:00 PM)
    time_steps: int = 13  # 30-minute intervals
    risk_aversion: float = 1.0  # Risk aversion parameter
    urgency: float = 1.0  # Urgency parameter (higher = more aggressive)

class ExecutionStrategy:
    """
    Implements optimal execution strategies to minimize temporary price impact.
    """
    
    def __init__(self, impact_model, order_book_snapshots: List):
        self.impact_model = impact_model
        self.order_book_snapshots = order_book_snapshots
        self.time_steps = len(order_book_snapshots)
        
    def calculate_impact_cost(self, order_sizes: List[float], snapshot_idx: int) -> float:
        """
        Calculate total impact cost for a sequence of orders.
        
        Args:
            order_sizes: List of order sizes for each time step
            snapshot_idx: Index of the order book snapshot to use
            
        Returns:
            Total impact cost
        """
        if snapshot_idx >= len(self.order_book_snapshots):
            snapshot_idx = len(self.order_book_snapshots) - 1
            
        snapshot = self.order_book_snapshots[snapshot_idx]
        self.impact_model.fit_piecewise_model(snapshot)
        
        total_cost = 0.0
        for size in order_sizes:
            if size > 0:
                impact = self.impact_model.predict_piecewise(size)
                total_cost += impact * size
                
        return total_cost
    
    def naive_strategy(self, total_shares: int, time_steps: int) -> List[float]:
        """
        Naive strategy: Execute equal amounts at each time step.
        
        Args:
            total_shares: Total shares to execute
            time_steps: Number of time steps
            
        Returns:
            List of order sizes for each time step
        """
        return [total_shares / time_steps] * time_steps
    
    def aggressive_strategy(self, total_shares: int, time_steps: int) -> List[float]:
        """
        Aggressive strategy: Execute more at the beginning.
        
        Args:
            total_shares: Total shares to execute
            time_steps: Number of time steps
            
        Returns:
            List of order sizes for each time step
        """
        # Exponential decay: more at beginning, less at end
        weights = np.exp(-np.arange(time_steps) * 0.3)
        weights = weights / np.sum(weights)
        return [total_shares * w for w in weights]
    
    def conservative_strategy(self, total_shares: int, time_steps: int) -> List[float]:
        """
        Conservative strategy: Execute more at the end.
        
        Args:
            total_shares: Total shares to execute
            time_steps: Number of time steps
            
        Returns:
            List of order sizes for each time step
        """
        # Exponential growth: less at beginning, more at end
        weights = np.exp(np.arange(time_steps) * 0.3)
        weights = weights / np.sum(weights)
        return [total_shares * w for w in weights]
    
    def optimal_strategy(self, total_shares: int, params: ExecutionParameters) -> Dict:
        """
        Optimal strategy using mathematical optimization.
        
        Args:
            total_shares: Total shares to execute
            params: Execution parameters
            
        Returns:
            Dictionary with optimal strategy results
        """
        time_steps = min(params.time_steps, len(self.order_book_snapshots))
        
        def objective_function(order_sizes):
            """Objective function to minimize: total impact cost + risk penalty."""
            total_cost = 0.0
            
            # Calculate impact cost for each time step
            for i, size in enumerate(order_sizes):
                if size > 0:
                    snapshot_idx = min(i, len(self.order_book_snapshots) - 1)
                    impact = self.calculate_impact_cost([size], snapshot_idx)
                    total_cost += impact
            
            # Add risk penalty for large orders
            risk_penalty = params.risk_aversion * np.sum(np.array(order_sizes) ** 2)
            
            # Add urgency penalty (encourages earlier execution)
            urgency_penalty = params.urgency * np.sum(np.array(order_sizes) * np.arange(len(order_sizes)))
            
            return total_cost + risk_penalty + urgency_penalty
        
        def constraint_function(order_sizes):
            """Constraint: total shares must equal target."""
            return np.sum(order_sizes) - total_shares
        
        # Initial guess: equal distribution
        initial_guess = [total_shares / time_steps] * time_steps
        
        # Bounds: non-negative order sizes
        bounds = [(0, total_shares)] * time_steps
        
        # Constraints: total shares constraint
        constraints = {'type': 'eq', 'fun': constraint_function}
        
        # Optimize
        result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_sizes = result.x
            total_cost = objective_function(optimal_sizes)
            
            return {
                'strategy': 'optimal',
                'order_sizes': optimal_sizes.tolist(),
                'total_cost': total_cost,
                'success': True,
                'message': result.message
            }
        else:
            return {
                'strategy': 'optimal',
                'order_sizes': initial_guess,
                'total_cost': objective_function(initial_guess),
                'success': False,
                'message': result.message
            }
    
    def compare_strategies(self, total_shares: int, params: ExecutionParameters) -> Dict:
        """
        Compare different execution strategies.
        
        Args:
            total_shares: Total shares to execute
            params: Execution parameters
            
        Returns:
            Dictionary with comparison results
        """
        time_steps = min(params.time_steps, len(self.order_book_snapshots))
        
        strategies = {
            'naive': self.naive_strategy(total_shares, time_steps),
            'aggressive': self.aggressive_strategy(total_shares, time_steps),
            'conservative': self.conservative_strategy(total_shares, time_steps)
        }
        
        # Add optimal strategy
        optimal_result = self.optimal_strategy(total_shares, params)
        strategies['optimal'] = optimal_result['order_sizes']
        
        # Calculate costs for each strategy
        costs = {}
        for name, sizes in strategies.items():
            total_cost = 0.0
            for i, size in enumerate(sizes):
                if size > 0:
                    snapshot_idx = min(i, len(self.order_book_snapshots) - 1)
                    impact = self.calculate_impact_cost([size], snapshot_idx)
                    total_cost += impact
            costs[name] = total_cost
        
        return {
            'strategies': strategies,
            'costs': costs,
            'optimal_result': optimal_result
        }
    
    def visualize_strategies(self, comparison_results: Dict, symbol: str):
        """
        Visualize different execution strategies.
        
        Args:
            comparison_results: Results from compare_strategies
            symbol: Stock symbol for title
        """
        strategies = comparison_results['strategies']
        costs = comparison_results['costs']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Order sizes over time
        time_steps = len(list(strategies.values())[0])
        x = np.arange(time_steps)
        
        colors = ['blue', 'red', 'green', 'purple']
        for i, (name, sizes) in enumerate(strategies.items()):
            ax1.plot(x, sizes, 'o-', color=colors[i], linewidth=2, 
                    label=f'{name.title()} (Cost: ${costs[name]:.4f})')
        
        ax1.set_xlabel('Time Step (30-min intervals)')
        ax1.set_ylabel('Order Size (shares)')
        ax1.set_title(f'Execution Strategies: {symbol}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cost comparison
        strategy_names = list(costs.keys())
        cost_values = list(costs.values())
        
        bars = ax2.bar(strategy_names, cost_values, color=colors[:len(strategy_names)], alpha=0.7)
        ax2.set_ylabel('Total Impact Cost ($)')
        ax2.set_title('Strategy Cost Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add cost labels on bars
        for bar, cost in zip(bars, cost_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(cost_values)*0.01,
                    f'${cost:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'execution_strategies_{symbol}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Main execution function for Part 2."""
    print("Execution Strategy Analysis: Optimal Order Splitting")
    print("=" * 60)
    
    # This would be integrated with the main market impact analysis
    # For demonstration, we'll create a simple example
    
    # Example parameters
    params = ExecutionParameters(
        total_shares=10000,
        trading_hours=6.5,
        time_steps=13,
        risk_aversion=0.001,
        urgency=0.1
    )
    
    print(f"Total shares to execute: {params.total_shares:,}")
    print(f"Trading hours: {params.trading_hours}")
    print(f"Time steps: {params.time_steps} (30-minute intervals)")
    print(f"Risk aversion: {params.risk_aversion}")
    print(f"Urgency: {params.urgency}")
    
    # Note: This would be integrated with the actual impact model and order book data
    # from the main analysis
    
    print("\nExecution strategies to implement:")
    print("1. Naive: Equal distribution across time")
    print("2. Aggressive: More execution at beginning")
    print("3. Conservative: More execution at end")
    print("4. Optimal: Mathematical optimization")
    
    print("\nNext steps:")
    print("- Integrate with market impact analysis")
    print("- Test with real order book data")
    print("- Compare strategy performance")
    print("- Generate comprehensive report")

if __name__ == "__main__":
    main() 