from time import perf_counter
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import animation
from queue_search_code import *

WALL, CHARGER, CLEAN, DIRTY = list(range(4))
SIZE = 7

class RoombaDomain:
    def __init__(self): 

        # deterministic grid world
        num_rows, num_cols = SIZE, SIZE
        grid = CLEAN*np.ones((num_rows, num_cols), dtype=int)
        grid[SIZE//2, 1:SIZE-1] = WALL
        grid[1:SIZE//2+1,SIZE//2] = WALL
        grid[0,0] = CHARGER
        grid[0,-1] = CHARGER
        grid[-1,SIZE//2] = CHARGER
        max_power = 2*SIZE + 1
        
        self.grid = grid
        self.max_power = max_power

    def pack(self, g, r, c, p):
        return (g.tobytes(), r, c, p)
    def unpack(self, state):
        grid, r, c, p = state
        grid = np.frombuffer(grid, dtype=int).reshape(self.grid.shape).copy()
        return grid, r, c, p

    def initial_state(self, roomba_position, dirty_positions):
        r, c = roomba_position
        grid = self.grid.copy()
        for dr, dc in dirty_positions: grid[dr, dc] = DIRTY
        return self.pack(grid, r, c, self.max_power)

    def render(self, ax, state, x=0, y=0):
        grid, r, c, p = self.unpack(state)
        num_rows, num_cols = grid.shape
        ax.imshow(grid, cmap='gray', vmin=0, vmax=3, extent=(x-.5,x+num_cols-.5, y+num_rows-.5, y-.5))
        for col in range(num_cols+1): pt.plot([x+ col-.5, x+ col-.5], [y+ -.5, y+ num_rows-.5], 'k-')
        for row in range(num_rows+1): pt.plot([x+ -.5, x+ num_cols-.5], [y+ row-.5, y+ row-.5], 'k-')
        pt.text(c-.25, r+.25, str(p), fontsize=24)
        pt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    def valid_actions(self, state):

        # r, c is the current row and column of the roomba
        # p is the current power level of the roomba
        # grid[i,j] is WALL, CHARGER, CLEAN or DIRTY to indicate status at row i, column j.
        grid, r, c, p = self.unpack(state)
        num_rows, num_cols = grid.shape
        actions = []
        ### TODO: Update the list of valid actions as described in the instruction PDF
        # actions[k] should have the form ((dr, dc), step_cost) for the kth valid action
        # where dr, dc are the change to roomba's row and column position
        if r > 0 and grid[r-1,c] != WALL: actions.append(((-1, 0), 1))
        if r < num_rows-1 and grid[r+1,c] != WALL: actions.append(((1, 0), 1))
        ### TODO: add code for column (c) boundaries
        if c > 0 and grid[r,c-1] > 0: actions.append(((0, -1), 1))
        if c < num_cols-1 and grid[r,c+1] != WALL: actions.append(((0, 1), 1))
        ### diagonal
        if c > 0 and r > 0 and grid[r-1,c-1] != WALL: actions.append(((-1,-1),1))
        if c < num_cols-1 and r < num_rows-1 and grid[r+1,c+1] != WALL: actions.append(((1,1),1))
        if c > 0 and r < num_rows-1 and grid[r+1,c-1] != WALL: actions.append(((1,-1),1))
        if c < num_cols-1 and r > 0 and grid[r-1,c+1] != WALL: actions.append(((-1,1),1))
        if grid[r,c]!=WALL: actions.append(((0,0),1))

        #if r == 0 or c== num_cols-1 or r == num_rows-1 and grid[r,c]!=WALL: actions.append(((0,0),1))
        ### TODO: add code to deal with zero power
        if p == 0: actions=[((0,0),1)]
        return actions
    
    def perform_action(self, state, action):
        grid, r, c, p = self.unpack(state)
        dr, dc = action

        if p>0 and (dr!=0 or dc!=0):
            p=p-1
            r, c = r + dr, c + dc
        else:    
            if grid[r,c] == CHARGER and p < self.max_power and (dr==0 and dc==0): p=p+1
        # TODO: update grid, r, c, and p as described in the instruction PDF
            if grid[r,c] == DIRTY:
                if (dr==0 and dc==0) and p>0: 
                    grid[r,c] = CLEAN
                    p=p-1

        new_state = self.pack(grid, r, c, p)
        return new_state

    def is_goal(self, state):
        grid, r, c, p = self.unpack(state)

        # In a goal state, no grid cell should be dirty
        result = (grid != DIRTY).all() and grid[r,c]==CHARGER
        ### TODO: Implement additional requirement that roomba is back at a charger

        return result

    def simple_heuristic(self, state):
        grid, r, c, p = self.unpack(state)

        # get list of dirty positions
        # dirty[k] has the form (i, j)
        # where (i, j) are the row and column position of the kth dirty cell
        dirty = list(zip(*np.nonzero(grid == DIRTY)))

        # if no positions are dirty, estimate zero remaining cost to reach a goal state
        if len(dirty) == 0: return 0

        # otherwise, get the distance from the roomba to each dirty square
        dists = [max(np.fabs(dr-r), np.fabs(dc-c)) for (dr, dc) in dirty]

        # estimate the remaining cost to goal as the largest distance to a dirty position
        return int(max(dists))

    def better_heuristic(self, state):

        ### TODO: Implement a "better" heuristic than simple_heuristic
        # "Better" means more memory-efficient (fewer popped nodes during A* search)
        grid, r, c, p = self.unpack(state)

        # get list of dirty positions
        # dirty[k] has the form (i, j)
        # where (i, j) are the row and column position of the kth dirty cell
        dirty = list(zip(*np.nonzero(grid == DIRTY)))

        # if no positions are dirty, estimate zero remaining cost to reach a goal state
        if len(dirty) == 0: return 0

        # otherwise, get the distance from the roomba to each dirty square
        #dists = [max(np.fabs(dr-r),np.fabs(dc-c),np.fabs(dr+c-1), np.fabs(dc+r-1)) for (dr, dc) in dirty]
        dists = [max(np.fabs(dr-r), np.fabs(dc-c),(int((np.fabs(dr-r)+np.fabs(dc-c)/1.7)))) for (dr, dc) in dirty]        #dists = [max((np.fabs(dr-r), np.fabs(dc-c),dr,dc,c,r,SIZE)) for (dr, dc) in dirty]
        # estimate the remaining cost to goal as the largest distance to a dirty position
        return int(max(dists))
        return 0

if __name__ == "__main__":


    # set up initial state by making five random open positions dirty
    domain = RoombaDomain()
    init = domain.initial_state(
        roomba_position = (0, 0),
        dirty_positions = np.random.permutation(list(zip(*np.nonzero(domain.grid == CLEAN))))[:5])

    problem = SearchProblem(domain, init, domain.is_goal)

    # compare runtime and node count for BFS, A* with simple heuristic, and A* with better heuristic
    start = perf_counter()
    plan, node_count = breadth_first_search(problem)
    bfs_time = perf_counter() - start
    print("bfs_time", bfs_time)
    print("node count", node_count)

    start = perf_counter()
    plan, node_count = a_star_search(problem, domain.simple_heuristic)
    astar_time = perf_counter() - start
    print("astar_time", astar_time)
    print("node count", node_count)

    start = perf_counter()
    plan, node_count = a_star_search(problem, domain.better_heuristic)
    astar_time = perf_counter() - start
    print("better heuristic:")
    print("astar_time", astar_time)
    print("node count", node_count)

    # reconstruct the intermediate states along the plan
    states = [problem.initial_state]
    for a in range(len(plan)):
        states.append(domain.perform_action(states[-1], plan[a]))

    # Animate the plan

    fig = pt.figure(figsize=(8,8))

    def drawframe(n):
        pt.cla()
        domain.render(pt.gca(), states[n])

    # blit=True re-draws only the parts that have changed.
    anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=500, blit=False)
    pt.show()

