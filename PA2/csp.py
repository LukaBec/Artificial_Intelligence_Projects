"""
CS311 Programming Assignment 2

Full Name(s): Luka Becerra

Brief description of my solver:
"""

import argparse, copy, time
from typing import Dict, List, Optional, Set, Tuple

# You are welcome to add constants, but do not modify the pre-existing constants

# Length of side of a Soduku board
SIDE = 9

# Length of side of "box" within a Soduku board
BOX = 3

# Domain for cells in Soduku board
DOMAIN = range(1, 10)

# Helper constant for checking a Soduku solution
SOLUTION = set(DOMAIN)


def check_solution(board: List[int], original_board: List[int]) -> bool:
    """Return True if board is a valid Sudoku solution to original_board puzzle"""
    # Original board values are maintained
    for s, o in zip(board, original_board):
        if o != 0 and s != o:
            return False
    for i in range(SIDE):
        # Valid row
        if set(board[i * SIDE : (i + 1) * SIDE]) != SOLUTION:
            return False
        # Valid column
        if set(board[i : SIDE * SIDE : SIDE]) != SOLUTION:
            return False
        # Valid Box
        box_row, box_col = (i // BOX) * BOX, (i % BOX) * BOX
        box = set()
        for r in range(box_row, box_row + BOX):
            box.update(board[r * SIDE + box_col : r * SIDE + box_col + BOX])
        if box != SOLUTION:
            return False
    return True


def backtracking_search(neighbors: List[List[int]], queue: Set[Tuple[int, int]], domains: List[List[int]]) -> Tuple[Optional[List[int]], int]:
    """Perform backtracking search on CSP using AC3

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable

    Returns:
        Tuple[Optional[List[int]], int]: Solution or None indicating no solution found and the number of recursive backtracking calls
    """
    # Track the number of recursive calls to backtrack
    recursions = 0

    # Defining a function within another creates a closure that has access to variables in the 
    # enclosing scope (e.g., to neighbors, etc). To be able to reassign those variables we use
    # the 'nonlocal' specification
    def backtrack(assignment: Dict[int, int]) -> Optional[Dict[int, int]]:
        """Backtrack search recursive function

        Args:
            assignment (Dict[int, int]): Values currently assigned to variables (variable index as key)

        Returns:
            Optional[Dict[int, int]]: Valid assignment or None if assignment is inconsistent
        """
        nonlocal recursions, domains # Enable us to reassign the recursions variable in the enclosing scope
        recursions += 1
    
        # Do we incorporate forward checking into our backtracker?
        """
        return assignment if complete
        choose variable x(i) in CSP
         - smartly order valiables and values
        Foreach value in D(i):  # D(i) is domain for x(i)
            if x(i) = value does no conflict with neighbors' assignment:
                Update assignment with x(i) <- value
                result <- backtrack(CSP, assignment)
                return result if result != failure
                Remove x(i) from assignment
        return failure
        """
        if len(assignment) == SIDE*SIDE:
            return assignment

        domains_copy = copy.deepcopy(domains)  
        if not ac3(domains):
          domains = domains_copy
          return None

        
        for var in range(SIDE*SIDE):
            if var not in assignment:
                break
 
        #for x in assignment:
        # if assignment[x] == domains[x]:
        #     continue
        #print(assignment)
        for val in domains[var][:]:
            #instead of finding first unassigned var, can look for domain with the least number of vars
            if not any(val == assignment[n] for n in neighbors[var] if n in assignment):
                assignment[var] = val
                domains[var] = [val]

                result = backtrack(assignment)
                if result is not None:
                    return result
                
                del assignment[var]

        domains = domains_copy
        return None

    def ac3(domains):# What do we do to implement AC3? Is it necessary?
        """
        function AC3(CSP):
            make a copy of the initial queue
            Enqueue all arcs (x(i), x(j)) and (x(j), x(i)) in queue Q
            while Q is not empty:
                (x(i), x(j)) <- pop(Q)
                if Revise(CSP, x(i), x(j)):
                    # Make D(i) consistent with D(j), returning True if D(i) is changed
                    # (making it smaller)
                    return False if D(i) is empty:
                    # Changing D(i) may enable further reductions in D(k), even if we
                    # had already considered x(k).
                    for x(k) in neighbors(X(i)) - {x(j)}:
                        add (x(k), x(i)) to Q
            return True
        """

        local_queue = queue.copy()
        while len(local_queue):
            xi, xj = local_queue.pop()
            if revise(domains, xi, xj):
                # Make D(i) consistent with D(j), returning True if D(i) is changed
                # (making it smaller)
                if len(domains[xi]) == 0:
                    return False
                for xk in neighbors[xi]:
                    if xk != xj:
                        local_queue.add((xk,xi))
        return True
        
    def revise(domains, xi, xj):
        """
        function Revise(CSP, x(i), x(j)):
            revised <- False
            for val in D(i):
                if no value in D(j) satisfies constraint (x(i), x(j)) when x(i) is val:
                    remove val from D(i)
                    revised <- True
            return revised
        """

        revised = False
        for val in domains[xi]:
            if len(domains[xj]) == 1 and domains[xj] == [val]:
                domains[xi].remove(val)
                revised = True
        return revised

    
    result = backtrack({var : domain[0] for var, domain in enumerate(domains) if len(domain) == 1})
    
    # Convert assignment dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, recursions


def sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using backtracking search with the AC3 algorithm

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and a count of calls to recursive backtracking function
    """
    
    domains = [[val] if val else list(DOMAIN) for val in board]
    neighbors = []
    queue = set()
    for i in range(SIDE*SIDE):
        my_neighbors = set()
        row, col= i // SIDE, i % SIDE
        my_neighbors.update(range(row*SIDE,(row+1)*SIDE))
        my_neighbors.update(range(col,SIDE*SIDE,SIDE))
        box_row, box_col = (row // BOX) * BOX, (col // BOX) * BOX
        for r in range(box_row, box_row + BOX):
            my_neighbors.update(range(r * SIDE + box_col, r * SIDE + box_col + BOX))
        neighbors.append(list(my_neighbors - {i}))
        for xj in neighbors[-1]:
            queue.add((i, xj))

    return backtracking_search(neighbors, queue, domains)
    


def my_sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using your own custom solver

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and a count of calls to recursive backtracking function
    """
    return None, 0


if __name__ == "__main__":
    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(description="Run sudoku solver")
    parser.add_argument(
        "-a",
        "--algo",
        default="ac3",
        help="Algorithm (one of ac3, custom)",
    )
    parser.add_argument(
        "-l",
        "--level",
        default="easy",
        help="Difficulty level (one of easy, medium, hard)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        type=int,
        help="Number of trials for timing",
    )
    parser.add_argument("puzzle", nargs="?", type=str, default=None)

    args = parser.parse_args()

    # fmt: off
    if args.puzzle:
        board = [int(c) for c in args.puzzle]
        if len(board) != SIDE*SIDE or set(board) > (set(DOMAIN) | { 0 }):
            raise ValueError("Invalid puzzle specification, it must be board length string with digits 0-9")
    elif args.level == "easy":
        board = [
            0,0,0,1,3,0,0,0,0,
            7,0,0,0,4,2,0,8,3,
            8,0,0,0,0,0,0,4,0,
            0,6,0,0,8,4,0,3,9,
            0,0,0,0,0,0,0,0,0,
            9,8,0,3,6,0,0,5,0,
            0,1,0,0,0,0,0,0,4,
            3,4,0,5,2,0,0,0,8,
            0,0,0,0,7,3,0,0,0,
        ]
    elif args.level == "medium":
        board = [
            0,4,0,0,9,8,0,0,5,
            0,0,0,4,0,0,6,0,8,
            0,5,0,0,0,0,0,0,0,
            7,0,1,0,0,9,0,2,0,
            0,0,0,0,8,0,0,0,0,
            0,9,0,6,0,0,3,0,1,
            0,0,0,0,0,0,0,7,0,
            6,0,2,0,0,7,0,0,0,
            3,0,0,8,4,0,0,6,0,
        ]
    elif args.level == "hard":
        board = [
            1,2,0,4,0,0,3,0,0,
            3,0,0,0,1,0,0,5,0,  
            0,0,6,0,0,0,1,0,0,  
            7,0,0,0,9,0,0,0,0,    
            0,4,0,6,0,3,0,0,0,    
            0,0,3,0,0,2,0,0,0,    
            5,0,0,0,8,0,7,0,0,    
            0,0,7,0,0,0,0,0,5,    
            0,0,0,0,0,0,0,9,8,
        ]
    else:
        raise ValueError("Unknown level")
    # fmt: on

    if args.algo == "ac3":
        solver = sudoku
    elif args.algo == "custom":
        solver = my_sudoku
    else:
        raise ValueError("Unknown algorithm type")

    times = []
    for i in range(args.trials):
        test_board = board[:] # Ensure original board is not modified
        start = time.perf_counter()
        solution, recursions = solver(test_board)
        end = time.perf_counter()
        times.append(end - start)
        if solution and not check_solution(solution, board):
            raise ValueError("Invalid solution")

        if solution:
            print(f"Trial {i} solved with {recursions} recursions")
            print(solution)
        else:
            print(f"Trial {i} not solved with {recursions} recursions")

    print(
        f"Minimum time {min(times)}s, Average time {sum(times) / args.trials}s (over {args.trials} trials)"
    )
