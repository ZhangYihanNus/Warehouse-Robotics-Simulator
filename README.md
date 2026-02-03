# Warehouse Robotics Simulator

This project simulates a warehouse environment with autonomous robots that pick up items from shelves and deliver them to stations. The simulation uses Python and features grid-based pathfinding, collision avoidance, and interactive controls.

## Features

- **Grid-based warehouse map**: Shelves, stations, and robots are placed on a 2D grid.
- **A* pathfinding**: Robots use the A* algorithm with Manhattan distance to navigate around obstacles.
- **Multi-robot coordination**: Robots avoid collisions and coordinate to complete tasks efficiently.
- **Task management**: Items are assigned to shelves and must be delivered to specific stations.
- **Interactive simulation**: Users can step through the simulation, add new tasks, and observe robot actions.

## How It Works

- Robots start at designated positions and wait for tasks.
- When a task is available, the nearest available robot is assigned to pick up the item from its shelf.
- The robot navigates to the shelf, picks up the item, and then delivers it to the target station.
- The simulation updates robot positions, item statuses, and displays the warehouse state at each tick.

## Running the Simulator

1. **Install dependencies**  
   Make sure you have Python 3 installed. You also need the `shortuuid` package:
   ```
   pip install shortuuid
   ```

2. **Run the simulation**  
   Execute the main Python file:
   ```
   python warehouse_robots_simulator.py
   ```

3. **Controls**  
   - Press `Enter` to advance the simulation by one tick.
   - Type `t <ShelfID> <StationID>` (e.g., `t 1 0`) to add a new delivery task.
   - Type `q` to quit the simulation.

## File Overview

- `warehouse_robots_simulator.py`: Main simulation logic, robot and item classes, pathfinding, and interactive loop.
- `README.md`: Project overview and instructions.

## Example

When you run the simulator, you'll see a grid representing the warehouse. Robots, shelves, and stations are displayed with different symbols. The dashboard shows the current tick, pending tasks, robot statuses, and item delivery progress.

---

Feel free to modify the warehouse layout, number of robots, or add new features!