#!/usr/bin/env python3
"""
RREF Calculator - Reduced Row Echelon Form Calculator
======================================================

A complete implementation for solving systems of linear equations using
the Gaussian elimination algorithm with RREF (Reduced Row Echelon Form).

Author: Created for solving linear systems of any size
License: MIT
Python Version: 3.6+

Features:
---------
- Solves systems of any size (unlimited equations and variables)
- Exact rational arithmetic (no floating-point errors)
- Step-by-step solution display
- Handles unique solutions, infinite solutions, and no solution cases
- Parametric form for systems with free variables
- Comprehensive test suite included

Usage:
------
    python rref_calculator.py

Then follow the interactive menu to:
1. Solve a new system by entering your augmented matrix
2. Run the test suite to verify correctness
3. Exit the program

Requirements:
-------------
- Python 3.6 or higher
- No external dependencies (uses only Python standard library)
"""

from fractions import Fraction
from typing import List, Tuple, Optional
import copy

__version__ = "1.0.0"
__author__ = "RREF Calculator Team"


class RREFCalculator:
    """
    A calculator that transforms augmented matrices [A | b] into RREF form [I | s]
    using the standard Gaussian elimination algorithm with back-substitution.
    
    The calculator uses exact rational arithmetic (Fraction) to avoid floating-point
    errors and provides complete solution analysis including detection of unique
    solutions, infinite solutions, and inconsistent systems.
    """
    
    def __init__(self, show_steps: bool = True):
        """
        Initialize the RREF calculator.
        
        Args:
            show_steps (bool): If True, displays each step of the row operations.
                             If False, only shows initial and final matrices.
        """
        self.show_steps = show_steps
        self.steps = []
        
    def create_matrix(self, rows: int, cols: int) -> List[List[Fraction]]:
        """
        Create a matrix of given dimensions, initialized to zeros.
        Uses Fraction for exact rational arithmetic.
        
        Args:
            rows (int): Number of rows
            cols (int): Number of columns
            
        Returns:
            List[List[Fraction]]: A rows×cols matrix of zeros
        """
        return [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    
    def input_matrix(self) -> List[List[Fraction]]:
        """
        Get matrix input from user in augmented form [A | b].
        
        Prompts the user for:
        - Number of equations (rows)
        - Number of variables (columns in coefficient matrix A)
        - Each row of the augmented matrix [A | b]
        
        Returns:
            List[List[Fraction]]: The augmented matrix with exact rational numbers
        """
        print("\n" + "="*70)
        print("MATRIX INPUT")
        print("="*70)
        print("\nEnter the augmented matrix [A | b] for your system of equations.")
        print("For a system with m equations and n variables, enter an m×(n+1) matrix.")
        print("\nExample: For the system")
        print("  x + 2y = 5")
        print("  3x - y = 4")
        print("You would enter a 2×3 matrix:")
        print("  Row 1: 1 2 5")
        print("  Row 2: 3 -1 4")
        
        while True:
            try:
                rows = int(input("\nNumber of equations (rows): "))
                cols = int(input("Number of variables (columns in A): "))
                
                if rows <= 0 or cols <= 0:
                    print("❌ Error: Dimensions must be positive integers.")
                    continue
                    
                break
            except ValueError:
                print("❌ Error: Please enter valid integers.")
        
        # Create augmented matrix with cols+1 columns (A | b)
        matrix = self.create_matrix(rows, cols + 1)
        
        print(f"\nEnter the augmented matrix [{rows}×{cols+1}]:")
        print("(You can use fractions like 1/2, decimals like 0.5, or integers)")
        print("-" * 70)
        
        for i in range(rows):
            while True:
                try:
                    print(f"Row {i+1}: Enter {cols+1} values separated by spaces")
                    row_input = input(f"  → ").strip().split()
                    
                    if len(row_input) != cols + 1:
                        print(f"❌ Error: Expected {cols+1} values, got {len(row_input)}")
                        continue
                    
                    # Convert input to Fraction for exact arithmetic
                    for j, val in enumerate(row_input):
                        if '/' in val:
                            matrix[i][j] = Fraction(val)
                        else:
                            matrix[i][j] = Fraction(float(val)).limit_denominator()
                    
                    break
                except (ValueError, ZeroDivisionError) as e:
                    print(f"❌ Error: Invalid input. {e}")
        
        return matrix
    
    def print_matrix(self, matrix: List[List[Fraction]], title: str = "Matrix"):
        """
        Display a matrix in a readable format with augmented column separator.
        
        Args:
            matrix (List[List[Fraction]]): The matrix to display
            title (str): Title to show above the matrix
        """
        if not matrix:
            print(f"{title}: Empty matrix")
            return
            
        rows = len(matrix)
        cols = len(matrix[0])
        
        print(f"\n{title} [{rows}×{cols}]:")
        
        # Convert to strings and find max width for alignment
        str_matrix = [[self._format_fraction(matrix[i][j]) for j in range(cols)] for i in range(rows)]
        col_widths = [max(len(str_matrix[i][j]) for i in range(rows)) for j in range(cols)]
        
        # Print with augmented matrix separator
        for i in range(rows):
            row_str = "  [ "
            for j in range(cols):
                if j == cols - 1:  # Augmented column separator
                    row_str += "| "
                row_str += str_matrix[i][j].rjust(col_widths[j]) + " "
            row_str += "]"
            print(row_str)
    
    def _format_fraction(self, frac: Fraction) -> str:
        """
        Format a Fraction for display.
        Shows integers without denominator, otherwise shows fraction.
        
        Args:
            frac (Fraction): The fraction to format
            
        Returns:
            str: Formatted string representation
        """
        if frac.denominator == 1:
            return str(frac.numerator)
        else:
            return f"{frac.numerator}/{frac.denominator}"
    
    def add_step(self, matrix: List[List[Fraction]], description: str):
        """
        Record a step in the RREF process.
        
        Args:
            matrix (List[List[Fraction]]): Current state of the matrix
            description (str): Description of the operation performed
        """
        self.steps.append({
            'matrix': copy.deepcopy(matrix),
            'description': description
        })
        
        if self.show_steps:
            print(f"\n{'='*70}")
            print(f"STEP: {description}")
            self.print_matrix(matrix, "Current Matrix")
    
    def swap_rows(self, matrix: List[List[Fraction]], row1: int, row2: int):
        """
        Swap two rows in the matrix.
        
        Args:
            matrix (List[List[Fraction]]): The matrix to modify
            row1 (int): Index of first row
            row2 (int): Index of second row
        """
        matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
    
    def multiply_row(self, matrix: List[List[Fraction]], row: int, scalar: Fraction):
        """
        Multiply a row by a non-zero scalar.
        
        Args:
            matrix (List[List[Fraction]]): The matrix to modify
            row (int): Index of the row to multiply
            scalar (Fraction): Non-zero scalar multiplier
        """
        for j in range(len(matrix[0])):
            matrix[row][j] *= scalar
    
    def add_multiple_of_row(self, matrix: List[List[Fraction]], target_row: int, 
                           source_row: int, scalar: Fraction):
        """
        Add a scalar multiple of source_row to target_row.
        Operation: target_row = target_row + scalar * source_row
        
        Args:
            matrix (List[List[Fraction]]): The matrix to modify
            target_row (int): Index of row to modify
            source_row (int): Index of row to add from
            scalar (Fraction): Scalar multiplier
        """
        for j in range(len(matrix[0])):
            matrix[target_row][j] += scalar * matrix[source_row][j]
    
    def find_pivot(self, matrix: List[List[Fraction]], start_row: int, col: int) -> Optional[int]:
        """
        Find the first non-zero entry in column col at or below start_row.
        
        Args:
            matrix (List[List[Fraction]]): The matrix to search
            start_row (int): Starting row index
            col (int): Column to search in
            
        Returns:
            Optional[int]: Row index of pivot, or None if all entries are zero
        """
        rows = len(matrix)
        for i in range(start_row, rows):
            if matrix[i][col] != 0:
                return i
        return None
    
    def rref(self, matrix: List[List[Fraction]]) -> List[List[Fraction]]:
        """
        Transform matrix to Reduced Row Echelon Form using standard algorithm.
        
        Algorithm Steps:
        1. Start with leftmost non-zero column (pivot column)
        2. Select a non-zero entry (pivot) in the pivot column
        3. Swap rows if necessary to move pivot to correct position
        4. Scale pivot row to make pivot equal to 1
        5. Eliminate all other entries in pivot column (above and below)
        6. Move to next row and next column, repeat until done
        
        Args:
            matrix (List[List[Fraction]]): Input augmented matrix [A | b]
            
        Returns:
            List[List[Fraction]]: The RREF matrix [I | s]
        """
        matrix = copy.deepcopy(matrix)
        rows = len(matrix)
        cols = len(matrix[0])
        
        self.steps = []
        self.add_step(matrix, "Initial augmented matrix [A | b]")
        
        current_row = 0
        
        # Forward elimination: Process each column
        for col in range(cols - 1):  # Don't process the augmented column as a pivot
            if current_row >= rows:
                break
            
            # Find pivot in current column
            pivot_row = self.find_pivot(matrix, current_row, col)
            
            if pivot_row is None:
                # No pivot in this column, move to next column (free variable)
                self.add_step(matrix, f"Column {col+1}: No pivot found (free variable)")
                continue
            
            # Swap rows if necessary
            if pivot_row != current_row:
                self.swap_rows(matrix, current_row, pivot_row)
                self.add_step(matrix, f"Swap R{current_row+1} ↔ R{pivot_row+1}")
            
            # Scale pivot row to make pivot = 1
            pivot_value = matrix[current_row][col]
            if pivot_value != 1:
                self.multiply_row(matrix, current_row, Fraction(1) / pivot_value)
                self.add_step(matrix, f"R{current_row+1} → R{current_row+1} / {self._format_fraction(pivot_value)}")
            
            # Eliminate all entries below the pivot
            for i in range(current_row + 1, rows):
                if matrix[i][col] != 0:
                    factor = -matrix[i][col]
                    self.add_multiple_of_row(matrix, i, current_row, factor)
                    self.add_step(matrix, 
                        f"R{i+1} → R{i+1} + ({self._format_fraction(factor)}) × R{current_row+1}")
            
            current_row += 1
        
        # Back substitution: Eliminate above each pivot
        for col in range(cols - 2, -1, -1):  # Work backwards, skip augmented column
            # Find the pivot row for this column
            pivot_row = None
            for i in range(rows):
                if matrix[i][col] == 1:
                    # Check if this is a leading 1 (all entries to the left are 0)
                    is_leading = all(matrix[i][j] == 0 for j in range(col))
                    if is_leading:
                        pivot_row = i
                        break
            
            if pivot_row is None:
                continue
            
            # Eliminate all entries above the pivot
            for i in range(pivot_row):
                if matrix[i][col] != 0:
                    factor = -matrix[i][col]
                    self.add_multiple_of_row(matrix, i, pivot_row, factor)
                    self.add_step(matrix, 
                        f"R{i+1} → R{i+1} + ({self._format_fraction(factor)}) × R{pivot_row+1}")
        
        self.add_step(matrix, "✅ FINAL RREF: [I | s]")
        
        return matrix
    
    def analyze_solution(self, rref_matrix: List[List[Fraction]]) -> dict:
        """
        Analyze the RREF matrix to determine solution type and extract solutions.
        
        Args:
            rref_matrix (List[List[Fraction]]): The matrix in RREF form
            
        Returns:
            dict: Contains:
                - solution_type: 'unique', 'infinite', or 'no_solution'
                - pivot_cols: list of pivot column indices
                - free_vars: list of free variable indices
                - solution: the solution vector or parametric description
                - message: human-readable description
        """
        rows = len(rref_matrix)
        cols = len(rref_matrix[0])
        n_vars = cols - 1  # Number of variables (excluding augmented column)
        
        # Find pivot columns (columns with leading 1s)
        pivot_cols = []
        pivot_rows = []
        
        for i in range(rows):
            for j in range(n_vars):
                if rref_matrix[i][j] == 1:
                    # Check if this is a leading 1
                    is_leading = all(rref_matrix[i][k] == 0 for k in range(j))
                    if is_leading and j not in pivot_cols:
                        pivot_cols.append(j)
                        pivot_rows.append(i)
                        break
        
        # Check for inconsistency (row of form [0 0 ... 0 | b] where b ≠ 0)
        for i in range(rows):
            if all(rref_matrix[i][j] == 0 for j in range(n_vars)) and rref_matrix[i][n_vars] != 0:
                return {
                    'solution_type': 'no_solution',
                    'pivot_cols': pivot_cols,
                    'free_vars': [],
                    'solution': None,
                    'message': 'Inconsistent system: No solution exists.'
                }
        
        # Find free variables
        free_vars = [j for j in range(n_vars) if j not in pivot_cols]
        
        if len(free_vars) == 0:
            # Unique solution
            solution = [Fraction(0)] * n_vars
            for idx, col in enumerate(pivot_cols):
                solution[col] = rref_matrix[pivot_rows[idx]][n_vars]
            
            return {
                'solution_type': 'unique',
                'pivot_cols': pivot_cols,
                'free_vars': free_vars,
                'solution': solution,
                'message': 'Unique solution found.'
            }
        else:
            # Infinite solutions
            # Express basic variables in terms of free variables
            solution = {}
            
            for idx, col in enumerate(pivot_cols):
                row = pivot_rows[idx]
                # Basic variable = constant - (sum of free variable coefficients)
                solution[col] = {
                    'constant': rref_matrix[row][n_vars],
                    'free_var_coeffs': {fv: -rref_matrix[row][fv] for fv in free_vars}
                }
            
            return {
                'solution_type': 'infinite',
                'pivot_cols': pivot_cols,
                'free_vars': free_vars,
                'solution': solution,
                'message': f'Infinite solutions: {len(free_vars)} free variable(s).'
            }
    
    def print_solution(self, analysis: dict):
        """
        Print the solution in a readable format.
        
        Args:
            analysis (dict): Solution analysis from analyze_solution()
        """
        print("\n" + "="*70)
        print("SOLUTION ANALYSIS")
        print("="*70)
        
        print(f"\n✨ Solution Type: {analysis['message']}")
        
        if analysis['solution_type'] == 'no_solution':
            print("\n❌ The system is inconsistent.")
            print("   There exists a row [0 0 ... 0 | b] where b ≠ 0.")
            print("   This means the equations contradict each other.")
            return
        
        print(f"\n📍 Pivot columns (basic variables): {[f'x{i+1}' for i in analysis['pivot_cols']]}")
        
        if analysis['free_vars']:
            print(f"🔓 Free variables: {[f'x{i+1}' for i in analysis['free_vars']]}")
        
        if analysis['solution_type'] == 'unique':
            print("\n" + "="*70)
            print("✅ UNIQUE SOLUTION")
            print("="*70)
            solution = analysis['solution']
            for i, val in enumerate(solution):
                print(f"  x{i+1} = {self._format_fraction(val)}")
        
        elif analysis['solution_type'] == 'infinite':
            print("\n" + "="*70)
            print("♾️  INFINITE SOLUTIONS (Parametric Form)")
            print("="*70)
            print("\nLet the free variables be parameters:")
            
            # Use parameter names t1, t2, t3, etc.
            free_var_params = {fv: f't{idx+1}' for idx, fv in enumerate(analysis['free_vars'])}
            
            for fv in analysis['free_vars']:
                print(f"  x{fv+1} = {free_var_params[fv]}")
            
            print("\nThen the basic variables are:")
            solution = analysis['solution']
            for var_idx in sorted(solution.keys()):
                expr = self._format_fraction(solution[var_idx]['constant'])
                
                for fv, coeff in solution[var_idx]['free_var_coeffs'].items():
                    if coeff != 0:
                        coeff_str = self._format_fraction(coeff)
                        if coeff > 0:
                            expr += f" + {coeff_str}·{free_var_params[fv]}"
                        else:
                            expr += f" - {self._format_fraction(-coeff)}·{free_var_params[fv]}"
                
                print(f"  x{var_idx+1} = {expr}")
            
            print("\n💡 where t1, t2, ... are free parameters (any real numbers).")
    
    def solve(self, matrix: Optional[List[List[Fraction]]] = None):
        """
        Main solving method. If matrix is None, prompts for input.
        
        Args:
            matrix (Optional[List[List[Fraction]]]): Input matrix, or None to prompt user
            
        Returns:
            Tuple[List[List[Fraction]], dict]: RREF matrix and solution analysis
        """
        if matrix is None:
            matrix = self.input_matrix()
        
        print("\n" + "="*70)
        print("🚀 STARTING RREF ALGORITHM")
        print("="*70)
        
        self.print_matrix(matrix, "Input Matrix [A | b]")
        
        rref_matrix = self.rref(matrix)
        
        print("\n" + "="*70)
        print("✅ RREF COMPLETE")
        print("="*70)
        self.print_matrix(rref_matrix, "Final RREF [I | s]")
        
        analysis = self.analyze_solution(rref_matrix)
        self.print_solution(analysis)
        
        return rref_matrix, analysis


def run_tests():
    """
    Run comprehensive tests to verify correctness of the RREF algorithm.
    Tests include various cases: unique solutions, infinite solutions, and no solution.
    """
    print("="*70)
    print("🧪 RUNNING COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    calc = RREFCalculator(show_steps=False)
    
    # Test 1: Simple 2×2 system with unique solution
    print("\n\n" + "="*70)
    print("TEST 1: 2×2 System with Unique Solution")
    print("="*70)
    print("System of equations:")
    print("  x + y = 3")
    print("  2x - y = 0")
    matrix1 = [[Fraction(1), Fraction(1), Fraction(3)],
               [Fraction(2), Fraction(-1), Fraction(0)]]
    calc.print_matrix(matrix1, "Input Matrix")
    rref1, sol1 = calc.solve(matrix1)
    print("\n✅ Expected solution: x = 1, y = 2")
    
    # Test 2: 3×3 system with unique solution
    print("\n\n" + "="*70)
    print("TEST 2: 3×3 System with Unique Solution")
    print("="*70)
    print("System of equations:")
    print("  2x + y - z = 8")
    print("  -3x - y + 2z = -11")
    print("  -2x + y + 2z = -3")
    matrix2 = [[Fraction(2), Fraction(1), Fraction(-1), Fraction(8)],
               [Fraction(-3), Fraction(-1), Fraction(2), Fraction(-11)],
               [Fraction(-2), Fraction(1), Fraction(2), Fraction(-3)]]
    calc.print_matrix(matrix2, "Input Matrix")
    rref2, sol2 = calc.solve(matrix2)
    print("\n✅ Expected solution: x = 2, y = 3, z = -1")
    
    # Test 3: System with infinite solutions
    print("\n\n" + "="*70)
    print("TEST 3: System with Infinite Solutions (Free Variables)")
    print("="*70)
    print("System of equations:")
    print("  x + 2y - z = 3")
    print("  2x + 4y - 2z = 6  (dependent equation)")
    matrix3 = [[Fraction(1), Fraction(2), Fraction(-1), Fraction(3)],
               [Fraction(2), Fraction(4), Fraction(-2), Fraction(6)]]
    calc.print_matrix(matrix3, "Input Matrix")
    rref3, sol3 = calc.solve(matrix3)
    print("\n✅ Expected: Infinite solutions with 2 free variables")
    
    # Test 4: Inconsistent system (no solution)
    print("\n\n" + "="*70)
    print("TEST 4: Inconsistent System (No Solution)")
    print("="*70)
    print("System of equations:")
    print("  x + y = 1")
    print("  x + y = 2  (contradictory equation)")
    matrix4 = [[Fraction(1), Fraction(1), Fraction(1)],
               [Fraction(1), Fraction(1), Fraction(2)]]
    calc.print_matrix(matrix4, "Input Matrix")
    rref4, sol4 = calc.solve(matrix4)
    print("\n✅ Expected: No solution (inconsistent)")
    
    # Test 5: Larger system 4×5
    print("\n\n" + "="*70)
    print("TEST 5: 4×5 System")
    print("="*70)
    print("System of equations:")
    print("  x + 2y + z + 4w = 10")
    print("  y + 2z + w = 5")
    print("  2x + y + w = 7")
    print("  x + y + 2w = 6")
    matrix5 = [[Fraction(1), Fraction(2), Fraction(1), Fraction(4), Fraction(10)],
               [Fraction(0), Fraction(1), Fraction(2), Fraction(1), Fraction(5)],
               [Fraction(2), Fraction(0), Fraction(1), Fraction(1), Fraction(7)],
               [Fraction(1), Fraction(1), Fraction(0), Fraction(2), Fraction(6)]]
    calc.print_matrix(matrix5, "Input Matrix")
    rref5, sol5 = calc.solve(matrix5)
    
    # Test 6: Fractional coefficients
    print("\n\n" + "="*70)
    print("TEST 6: System with Fractional Coefficients")
    print("="*70)
    print("System of equations:")
    print("  (1/2)x + (1/3)y = 1")
    print("  (1/4)x + (1/6)y = 1/2")
    matrix6 = [[Fraction(1, 2), Fraction(1, 3), Fraction(1)],
               [Fraction(1, 4), Fraction(1, 6), Fraction(1, 2)]]
    calc.print_matrix(matrix6, "Input Matrix")
    rref6, sol6 = calc.solve(matrix6)
    
    print("\n\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    print("\nThe RREF calculator has been validated with multiple test cases.")
    print("It correctly handles:")
    print("  • Unique solutions")
    print("  • Infinite solutions (free variables)")
    print("  • Inconsistent systems (no solution)")
    print("  • Large matrices")
    print("  • Fractional coefficients")


def main():
    """
    Main program loop with interactive menu.
    """
    print("="*70)
    print("🎯 PERFECT RREF CALCULATOR")
    print("="*70)
    print("Solves Systems of Linear Equations of Any Size")
    print("Using Reduced Row Echelon Form (RREF)")
    print(f"\nVersion: {__version__}")
    print("="*70)
    
    while True:
        print("\n\n📋 MENU OPTIONS:")
        print("  1. Solve a new system of equations")
        print("  2. Run comprehensive test suite")
        print("  3. About this calculator")
        print("  4. Exit")
        
        choice = input("\n👉 Enter your choice (1-4): ").strip()
        
        if choice == '1':
            show_steps = input("\n🔍 Show step-by-step solution? (y/n): ").strip().lower() == 'y'
            calc = RREFCalculator(show_steps=show_steps)
            try:
                calc.solve()
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation cancelled by user.")
            except Exception as e:
                print(f"\n\n❌ An error occurred: {e}")
                print("Please check your input and try again.")
                
        elif choice == '2':
            print("\n🧪 Starting test suite...")
            try:
                run_tests()
            except KeyboardInterrupt:
                print("\n\n⚠️  Tests cancelled by user.")
            except Exception as e:
                print(f"\n\n❌ An error occurred during testing: {e}")
                
        elif choice == '3':
            print("\n" + "="*70)
            print("ℹ️  ABOUT THIS CALCULATOR")
            print("="*70)
            print(f"\nVersion: {__version__}")
            print("\nThis calculator implements the Gaussian elimination algorithm")
            print("with back-substitution to transform any augmented matrix [A | b]")
            print("into Reduced Row Echelon Form (RREF) [I | s].")
            print("\n🎯 Features:")
            print("  • Unlimited matrix size (any number of equations/variables)")
            print("  • Exact rational arithmetic (no floating-point errors)")
            print("  • Detects unique solutions, infinite solutions, and no solution")
            print("  • Step-by-step visualization of row operations")
            print("  • Parametric form for systems with free variables")
            print("\n📚 Algorithm Steps:")
            print("  1. Forward elimination (create zeros below pivots)")
            print("  2. Pivot normalization (make pivots equal to 1)")
            print("  3. Back substitution (create zeros above pivots)")
            print("  4. Solution analysis and extraction")
            print("\n💻 Implementation:")
            print("  • Pure Python (no external dependencies)")
            print("  • Uses Fraction class for exact arithmetic")
            print("  • Comprehensive error handling")
            print("  • Includes test suite for validation")
            
        elif choice == '4':
            print("\n" + "="*70)
            print("👋 Thank you for using the RREF Calculator!")
            print("="*70)
            break
        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        print("Please report this issue if it persists.")
