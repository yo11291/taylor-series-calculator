"""
Taylor Series Calculator
A general-purpose Python program for calculating Taylor series approximations
of mathematical functions around a given point.
"""

import math
from typing import Callable, List, Tuple


class TaylorSeriesCalculator:
    """
    A class to calculate Taylor series approximations of functions.
    
    The Taylor series approximates a function f(x) around a point a as:
    f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + f'''(a)(x-a)³/3! + ...
    """
    
    def __init__(self, center: float = 0.0, num_terms: int = 10):
        """
        Initialize the Taylor series calculator.
        
        Args:
            center: The point around which to expand the series (default: 0)
            num_terms: Number of terms to include in the approximation (default: 10)
        """
        self.center = center
        self.num_terms = num_terms
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial of n."""
        if n <= 1:
            return 1
        return n * TaylorSeriesCalculator.factorial(n - 1)
    
    @staticmethod
    def numerical_derivative(func: Callable, x: float, h: float = 1e-7) -> float:
        """
        Calculate numerical derivative using central difference method.
        
        Args:
            func: The function to differentiate
            x: The point at which to calculate the derivative
            h: The step size for approximation
        
        Returns:
            The numerical derivative at x
        """
        return (func(x + h) - func(x - h)) / (2 * h)
    
    def get_derivatives(self, func: Callable, order: int) -> List[float]:
        """
        Get derivatives of a function at the center point.
        
        Args:
            func: The function to differentiate
            order: The maximum derivative order to calculate
        
        Returns:
            A list of derivatives [f(a), f'(a), f''(a), ...]
        """
        derivatives = [func(self.center)]
        current_func = func
        
        for _ in range(order):
            derivative_at_center = self.numerical_derivative(current_func, self.center)
            derivatives.append(derivative_at_center)
            
            # Create a new function for the next derivative
            prev_func = current_func
            current_func = lambda x, f=prev_func: self.numerical_derivative(f, x)
        
        return derivatives
    
    def approximate(self, func: Callable, x: float) -> float:
        """
        Calculate Taylor series approximation of func at point x.
        
        Args:
            func: The function to approximate
            x: The point at which to evaluate the approximation
        
        Returns:
            The Taylor series approximation value
        """
        derivatives = self.get_derivatives(func, self.num_terms)
        result = 0.0
        
        for n in range(self.num_terms):
            term = derivatives[n] * ((x - self.center) ** n) / self.factorial(n)
            result += term
        
        return result
    
    def error(self, func: Callable, x: float) -> float:
        """
        Calculate the error between actual and approximated values.
        
        Args:
            func: The actual function
            x: The point at which to evaluate
        
        Returns:
            The absolute error
        """
        actual = func(x)
        approx = self.approximate(func, x)
        return abs(actual - approx)


# Pre-built Taylor series for common functions (more efficient)
class CommonFunctions:
    """Optimized Taylor series approximations for common mathematical functions."""
    
    @staticmethod
    def sin(x: float, center: float = 0.0, num_terms: int = 10) -> float:
        """Taylor series approximation of sin(x)."""
        result = 0.0
        for n in range(num_terms):
            term = ((-1) ** n) * ((x - center) ** (2*n + 1)) / math.factorial(2*n + 1)
            result += term
        return result
    
    @staticmethod
    def cos(x: float, center: float = 0.0, num_terms: int = 10) -> float:
        """Taylor series approximation of cos(x)."""
        result = 0.0
        for n in range(num_terms):
            term = ((-1) ** n) * ((x - center) ** (2*n)) / math.factorial(2*n)
            result += term
        return result
    
    @staticmethod
    def exp(x: float, center: float = 0.0, num_terms: int = 10) -> float:
        """Taylor series approximation of e^x."""
        result = 0.0
        for n in range(num_terms):
            term = ((x - center) ** n) / math.factorial(n)
            result += term
        return result
    
    @staticmethod
    def ln(x: float, center: float = 1.0, num_terms: int = 10) -> float:
        """Taylor series approximation of ln(x) around x=1."""
        result = 0.0
        for n in range(1, num_terms):
            term = ((-1) ** (n + 1)) * ((x - center) ** n) / n
            result += term
        return result


def main():
    """Demonstrate Taylor series calculator with examples."""
    
    print("=" * 60)
    print("Taylor Series Calculator")
    print("=" * 60)
    
    # Example 1: sin(x)
    print("\n1. sin(x) approximation around x=0:")
    print("-" * 60)
    x_val = math.pi / 4  # 45 degrees
    actual_sin = math.sin(x_val)
    approx_sin = CommonFunctions.sin(x_val, num_terms=10)
    print(f"   x = π/4 ≈ {x_val:.4f}")
    print(f"   Actual sin(π/4) = {actual_sin:.10f}")
    print(f"   Taylor approx (10 terms) = {approx_sin:.10f}")
    print(f"   Error = {abs(actual_sin - approx_sin):.2e}")
    
    # Example 2: cos(x)
    print("\n2. cos(x) approximation around x=0:")
    print("-" * 60)
    actual_cos = math.cos(x_val)
    approx_cos = CommonFunctions.cos(x_val, num_terms=10)
    print(f"   x = π/4 ≈ {x_val:.4f}")
    print(f"   Actual cos(π/4) = {actual_cos:.10f}")
    print(f"   Taylor approx (10 terms) = {approx_cos:.10f}")
    print(f"   Error = {abs(actual_cos - approx_cos):.2e}")
    
    # Example 3: e^x
    print("\n3. e^x approximation around x=0:")
    print("-" * 60)
    x_val = 1.0
    actual_exp = math.exp(x_val)
    approx_exp = CommonFunctions.exp(x_val, num_terms=10)
    print(f"   x = {x_val}")
    print(f"   Actual e^{x_val} = {actual_exp:.10f}")
    print(f"   Taylor approx (10 terms) = {approx_exp:.10f}")
    print(f"   Error = {abs(actual_exp - approx_exp):.2e}")
    
    # Example 4: ln(x)
    print("\n4. ln(x) approximation around x=1:")
    print("-" * 60)
    x_val = 2.0
    actual_ln = math.log(x_val)
    approx_ln = CommonFunctions.ln(x_val, num_terms=15)
    print(f"   x = {x_val}")
    print(f"   Actual ln({x_val}) = {actual_ln:.10f}")
    print(f"   Taylor approx (15 terms) = {approx_ln:.10f}")
    print(f"   Error = {abs(actual_ln - approx_ln):.2e}")
    
    # Example 5: Custom function using numerical derivatives
    print("\n5. Custom function: f(x) = x³ around x=0:")
    print("-" * 60)
    custom_func = lambda x: x**3
    calc = TaylorSeriesCalculator(center=0.0, num_terms=5)
    x_val = 2.0
    actual = custom_func(x_val)
    approx = calc.approximate(custom_func, x_val)
    error = calc.error(custom_func, x_val)
    print(f"   x = {x_val}")
    print(f"   Actual f({x_val}) = {actual:.10f}")
    print(f"   Taylor approx (5 terms) = {approx:.10f}")
    print(f"   Error = {error:.2e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
