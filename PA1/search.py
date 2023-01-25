"""
CS311 Programming Assignment 1

Full Name(s): Luka Becerra

Brief description of my heuristic:
"""

import argparse, itertools, random
from math import inf
from typing import Callable, List, Optional, Sequence, Tuple
from heapq import heappush, heappop


# You are welcome to add constants, but do not modify the pre-existing constants

# Problem size 
BOARD_SIZE = 3

# The goal is a "blank" (0) in bottom right corner
GOAL = tuple(range(1, BOARD_SIZE**2)) + (0,)


def inversions(board: Sequence[int]) -> int:
    """Return the number of times a larger 'piece' precedes a 'smaller' piece in board"""
    return sum(
        (a > b and a != 0 and b != 0) for (a, b) in itertools.combinations(board, 2)
    )


class Node:
    def __init__(self, state: Sequence[int], parent: "Node" = None, cost=0):
        """Create Node to track particular state and associated parent and cost

        State is tracked as a "row-wise" sequence, i.e., the board (with _ as the blank)
        1 2 3
        4 5 6
        7 8 _
        is represented as (1, 2, 3, 4, 5, 6, 7, 8, 0) with the blank represented with a 0

        Args:
            state (Sequence[int]): State for this node, typically a list, e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
            parent (Node, optional): Parent node, None indicates the root node. Defaults to None.
            cost (int, optional): Cost in moves to reach this node. Defaults to 0.
        """
        self.state = tuple(state)  # To facilitate "hashable" make state immutable
        self.parent = parent
        self.cost = cost

    def is_goal(self) -> bool:
        """Return True if Node has goal state"""
        return self.state == GOAL

    def expand(self) -> List["Node"]:
        """Expand current node into possible child nodes with corresponding parent and cost"""

        # TODO: Implement this function to generate child nodes based on the current state
        children = []
        row = self.state.index(0) // 3
        col = self.state.index(0) % 3
        if row < 2:
            children.append(Node(self._swap(row,col,row+1,col),self, self.cost+1))
        if row > 0:
            children.append(Node(self._swap(row,col,row-1,col),self, self.cost+1))
        if col < 2:
            children.append(Node(self._swap(row,col,row,col+1),self, self.cost+1))
        if col > 0:
            children.append(Node(self._swap(row,col,row,col-1),self, self.cost+1))
        return children

    def expand_old(self) -> List["Node"]:
        """Expand current node into possible child nodes with corresponding parent and cost"""
        curr = self.state.index(0)
        if curr == 4:
            #checked
            children = [Node(self._swap(0,1,1,1),self,self.cost+1), Node(self._swap(1,0,1,1),self,self.cost+1), Node(self._swap(1,2,1,1),self,self.cost+1), Node(self._swap(2,1,1,1),self,self.cost+1)]
        elif curr % 2 == 1:
            if curr == 1:
                #checked
                children = [Node(self._swap(0,0,0,1),self,self.cost+1), Node(self._swap(0,2,0,1),self,self.cost+1), Node(self._swap(1,1,0,1),self,self.cost+1)]
            elif curr == 3:
                #checked
                children = [Node(self._swap(0,0,1,0),self,self.cost+1), Node(self._swap(1,1,1,0),self,self.cost+1), Node(self._swap(2,0,1,0),self,self.cost+1)]
            elif curr == 5:
                #checked
                children = [Node(self._swap(0,2,1,2),self,self.cost+1), Node(self._swap(1,1,1,2),self,self.cost+1), Node(self._swap(2,2,1,2),self,self.cost+1)]
            else:
                #checked
                children = [Node(self._swap(2,0,2,1),self,self.cost+1), Node(self._swap(1,1,2,1),self,self.cost+1), Node(self._swap(2,2,2,1),self,self.cost+1)]
        else:
            if curr == 0:
                #checked
                children = [Node(self._swap(0,1,0,0),self,self.cost+1), Node(self._swap(1,0,0,0),self,self.cost+1)]
            elif curr == 2:
                #checked
                children = [Node(self._swap(0,1,0,2),self,self.cost+1), Node(self._swap(1,2,0,2),self,self.cost+1)]
            elif curr == 6:
                #redone
                children = [Node(self._swap(2,1,2,0),self,self.cost+1), Node(self._swap(1,0,2,0),self,self.cost+1)]
            else:
                #checked
                children = [Node(self._swap(2,1,2,2),self,self.cost+1), Node(self._swap(1,2,2,2),self,self.cost+1)]
        # Node(...,self,self.cost+1)
        return children

    def _swap(self, row1: int, col1: int, row2: int, col2: int) -> Sequence[int]:
        """Swap values in current state bewteen row1,col1 and row2,col2, returning new "state" to construct a Node"""
        state = list(self.state)
        state[row1 * BOARD_SIZE + col1], state[row2 * BOARD_SIZE + col2] = (
            state[row2 * BOARD_SIZE + col2],
            state[row1 * BOARD_SIZE + col1],
        )
        return state

    def __str__(self):
        return str(self.state)

    # The following methods enable Node to be used in types that use hashing (sets, dictionaries) or perform comparisons. Note
    # that the comparisons are performed exclusively on the state and ignore parent and cost values.

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


def bfs(initial_board: Sequence[int], max_depth=12) -> Tuple[Optional[Node], int]:
    """Perform breadth-first search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    frontier,explored = [], []
    #What do I use for placeholder solution so I can compare in while loop?
    #Should I use a while loop?
    #Break down how the GOAL const works - what is read through the tuple
    start =  Node(initial_board)

    if initial_board == GOAL:
        return start, 1
    
    frontier.append(start)
    explored = {start}
    while len(frontier) != 0:
        #where to include depth
        curr = frontier.pop(0)
        explored.add(curr)
        if curr.state == GOAL:
            return curr, len(explored)
        if curr.cost >= max_depth:
            continue
        children = curr.expand()
        for child in children:
            if child in explored:
                continue
            else:
                frontier.append(child)

    return  None, len(explored)

def manhattan_distance(node: Node) -> int:
    """Compute manhattan distance f(node), i.e., g(node) + h(node)"""
    #Look into how MH dist composes heuristic
    h = 0
    for i, item in enumerate(node.state):
        if item != 0:
            row = i // 3
            col = i % 3 
            g_row = GOAL.index(item) // 3
            g_col = GOAL.index(item) % 3
            h += abs(row-g_row) + abs(col - g_col)
    dist = node.cost + h
    return dist

def custom_heuristic(node: Node) -> int:
    dist = manhattan_distance(node)
    if node.is_goal():
        pass
    elif node.state.index(6) // 3 != 2 and node.state.index(8) % 3 != 2:
        dist += 2
    return dist

def min_heap(nodes: Sequence[int], node: int) -> Sequence[int]:
    left = (2 * node) + 1
    right = (2 * node) + 2
    if left < len(nodes) and nodes[left] < nodes[node]:
        smallest = left
    else: smallest = node
    if right < len(nodes) and nodes[right] < nodes[node]:
        smallest = right
    #not leaf node
    if smallest != node:
        nodes[node], nodes[smallest] = nodes[smallest], nodes[node] #swap
        min_heap(nodes, smallest)

def build_heap(nodes: Sequence[int]) -> Sequence[int]:
    n = (len(nodes)//2 - 1)
    for node in range(n, -1. -1):
        min_heap(nodes,node)
    
def eval(node: Node) -> int:
    f_value = -inf
    return f_value

def astar(
    initial_board: Sequence[int],
    max_depth=12,
    heuristic: Callable[[Node], int] = custom_heuristic,
) -> Tuple[Optional[Node], int]:
    """Perform astar search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.
        heuristic (_Callable[[Node], int], optional): Heuristic function. Defaults to manhattan_distance.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored

    A*(initNode)
    1. Initialize a Priority Queue (e.g., Min-Binary Heap) to represent the Frontier. Initialize a separate data structure for the Reached nodes that supports efficient queries by state.
    2. Set the f(n)-value for initNode (recall that f(n) = g(n) + h(n)). For the initial node, g(n) = 0 and h(n) is the sum of the Manhattan Distance to reach the goal state. Add initNode to Frontier. 
    3. While the goal has not been found and Frontier is not empty and cost <= 12 :
        o Remove the node with the lowest f(n)-value from Frontier. Denote this node X. 
        o If X is the goal state, return X and the number of nodes reached.
        o Otherwise, expand X: 
            For each child node, C, of X:
            o If C is in Reached with the same or lower cost (i.e., g(n)), ignore it. 
            o Otherwise, compute the f(n)-value for C. For each C, g(n) is the cost to get to C (this should be 1 more than the g(n)-value for X) and h(n) is the Manhattan Distances to reach the goal from C. 
            o Add C to Frontier and add or update C in Reached (if cost is lower)
    4.    No solution has been found, return None and the number nodes reached
    """
    reached = {}
    frontier = []
    start =  Node(initial_board)
    heappush(frontier,(heuristic(start),start))

    #set f(n) value - WHAT IS INITNODE?
    while len(frontier) != 0:
        score, node = heappop(frontier)
        if node.cost > max_depth:
            break
        elif node.is_goal():
            return node, len(reached)
        children = node.expand()
        for child in children:
            if child.state not in reached:
                reached[child.state] = child
                heappush(frontier, (heuristic(child), child))
            elif reached[child.state].cost > child.cost:
                    reached[child.state] = child
                    heappush(frontier, (heuristic(child), child))


        pass
    return None, len(reached)

if __name__ == "__main__":

    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(
        description="Run search algorithms in random inputs"
    )
    parser.add_argument(
        "-a",
        "--algo",
        default="bfs",
        help="Algorithm (one of bfs, astar, astar_custom)",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=1000,
        help="Number of iterations",
    )
    parser.add_argument(
        "-s",
        "--state",
        type=str,
        default=None,
        help="Execute a single iteration using this board configuration specified as a string, e.g., 123456780",
    )

    args = parser.parse_args()

    num_solutions = 0
    num_cost = 0
    num_nodes = 0

    if args.algo == "bfs":
        algo = bfs
    elif args.algo == "astar":
        algo = astar
    elif args.algo == "astar_custom":
        algo = lambda board: astar(board, heuristic=custom_heuristic)
    else:
        raise ValueError("Unknown algorithm type")

    if args.state is None:
        iterations = args.iter
        while iterations > 0:
            init_state = list(range(BOARD_SIZE**2))
            random.shuffle(init_state)

            # A problem is only solvable if the parity of the initial state matches that
            # of the goal.
            if inversions(init_state) % 2 != inversions(GOAL) % 2:
                continue

            solution, nodes = algo(init_state)
            if solution:
                num_solutions += 1
                num_cost += solution.cost
                num_nodes += nodes

            iterations -= 1
    else:
        # Attempt single input state
        solution, nodes = algo([int(s) for s in args.state])
        if solution:
            num_solutions = 1
            num_cost = solution.cost
            num_nodes = nodes

    if num_solutions:
        print(
            "Iterations:",
            args.iter,
            "Solutions:",
            num_solutions,
            "Average moves:",
            num_cost / num_solutions,
            "Average nodes:",
            num_nodes / num_solutions,
        )
    else:
        print("Iterations:", args.iter, "Solutions: 0")
