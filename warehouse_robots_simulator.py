import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional
from enum import Enum
import heapq
import shortuuid

# --- 1. Basic Definitions ---

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)

class ItemStatus(Enum):
    PENDING = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    UNREACHABLE = 3

@dataclass
class Item:
    uuid: str
    shelf_id: int
    target_station_id: int
    status: ItemStatus = ItemStatus.PENDING

@dataclass
class Shelf:
    id: int
    x: int
    y: int
    item_uuid: Optional[str]

@dataclass
class Station:
    id: int
    x: int
    y: int

@dataclass
class RobotSnapshot:
    id: int
    x: int
    y: int
    carrying_item_uuid: Optional[str] 
    target: Optional[Tuple[int, int]] 


# Map Configuration
GRID_WIDTH = 20
GRID_HEIGHT = 10

# --- 2. Pathfinding and Robot Control ---

def heuristic(a, b):
    """Manhattan distance heuristic for A* pathfinding."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_path(start: tuple, end: tuple, shelves: List[Shelf]) -> list:
    """
    A* pathfinding algorithm using Manhattan distance heuristic.
    
    Args:
        start: (x, y) starting coordinates
        end: (x, y) target coordinates
        shelves: list of obstacles on the map

    Returns:
        List of coordinates from start to end (inclusive), or empty list if no path exists.
    """
    
    # Initialize data structures
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    # Main A* loop
    while open_list:
        current_f, current = heapq.heappop(open_list)

        # Target reached - reconstruct path
        if current == end:
            path = []
            key_pos = current
            while key_pos in came_from:
                path.append(key_pos)
                key_pos = came_from[key_pos]
            path.append(start)
            return path[::-1]

        # Explore neighbors (up, down, left, right)
        neighbors = [
            (current[0], current[1] - 1),  # up
            (current[0], current[1] + 1),  # down
            (current[0] - 1, current[1]),  # left
            (current[0] + 1, current[1])   # right
        ]

        for neighbor in neighbors:
            nx, ny = neighbor
            
            # Check grid boundaries
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue
            
            # Check for obstacles (allow target position even if occupied)
            if any(neighbor == (shelf.x, shelf.y) for shelf in shelves) and neighbor != end:
                continue

            # Calculate new cost
            tentative_g = g_score[current] + 1

            # Update path if this route is better
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, end)
                f_score[neighbor] = f
                
                # Add to open list if not already present
                if not any(neighbor == item[1] for item in open_list):
                    heapq.heappush(open_list, (f, neighbor))

    # No path found
    return []

class RobotPhysics:
    def __init__(self, robot_id, start_x, start_y, shelves: Dict[str, Shelf], stations: List[Station]):
        self.id = robot_id
        self.x = start_x
        self.y = start_y
        self.carrying_item_uuid = None
        self.shelves = shelves
        self.stations = stations
        self.target = None
    
    def get_snapshot(self) -> RobotSnapshot:
        return RobotSnapshot(
            id=self.id,
            x=self.x,
            y=self.y,
            carrying_item_uuid=self.carrying_item_uuid,
            target=self.target
        )

    def tick(self, action: Direction):
        """
        Update robot position based on action.
        - Move to new position
        - Enforce grid boundaries
        - Update carried shelf coordinates
        """
        next_pos = (self.x + action.value[0], self.y + action.value[1])
        self.x, self.y = next_pos


class GroupRobotAlgorithm:
    def __init__(self, map_width, map_height, stations: List[Station]):
        self.width = map_width
        self.height = map_height
        self.stations_map = {s.id: s for s in stations}

    def position_is_valid(self, position: Tuple[int, int], robot_snapshots: List[RobotSnapshot], shelves: Dict[str, Shelf]) -> bool:
        """Check if position is within bounds, unoccupied, and not blocking an empty shelf."""
        x, y = position

        if 0 <= x and x < GRID_WIDTH \
        and 0 <= y and y < GRID_HEIGHT \
        and (x, y) not in [(r.x, r.y) for r in robot_snapshots]:
            for shelf in shelves.values():
                if (shelf.x, shelf.y) == position and shelf.item_uuid is None:
                    return False
            return True
        return False

    def decide_next_action(self, robot_snapshots: List[RobotSnapshot], shelves: Dict[str, Shelf]) -> List[Direction]:
        """
        Compute next action for each robot based on current state and tasks.
        
        Returns:
            List of Direction actions matching robot order.
        """
        actions = []

        for robot in robot_snapshots:
            if robot.target is None:
                actions.append(Direction.STAY)
                continue

            path = find_path(
                start=(robot.x, robot.y), 
                end=robot.target, 
                shelves=list(shelves.values())
            ) 
            if path == []:
                print(f"No path found. Robot ID: {robot.id} | Position: ({robot.x}, {robot.y}) -> Target: {robot.target}")
                next_dir = Direction.STAY
            elif len(path) == 1:
                next_dir = Direction.STAY
            else:
                next_step = path[1]
                if not self.position_is_valid(next_step, robot_snapshots, shelves):
                    print(f"Collision detected. Robot ID: {robot.id} | Position: ({robot.x}, {robot.y}) -> Next: {next_step} -> Target: {robot.target}")
                    next_dir = Direction.STAY
                else:
                    dx = next_step[0] - robot.x
                    dy = next_step[1] - robot.y
                    next_dir = Direction((dx, dy))

            actions.append(next_dir) 
        return actions

# --- 3. Visualization & Simulation ---

class InteractiveSimulator:

    def __init__(self):
        self.stations = [
            Station(id=0, x=0, y=0),
            Station(id=1, x=0, y=GRID_HEIGHT-1)
        ]
        self.items = [
            Item(shelf_id=1, uuid=shortuuid.ShortUUID().random(length=8), target_station_id=0),
            Item(shelf_id=4, uuid=shortuuid.ShortUUID().random(length=8), target_station_id=1),
            Item(shelf_id=7, uuid=shortuuid.ShortUUID().random(length=8), target_station_id=0),
            Item(shelf_id=9, uuid=shortuuid.ShortUUID().random(length=8), target_station_id=1),
            Item(shelf_id=12, uuid=shortuuid.ShortUUID().random(length=8), target_station_id=0)
        ]
        
        # Initialize shelves with items
        self.shelves = {}
        s_id = 0
        for y in range(2, 4):
            for x in range(6, GRID_WIDTH-5):
                if s_id in [i.shelf_id for i in self.items]:
                    item = next(i for i in self.items if i.shelf_id == s_id)
                    self.shelves[s_id] = Shelf(id=s_id, x=x, y=y, item_uuid=item.uuid)
                else:
                    self.shelves[s_id] = Shelf(id=s_id, x=x, y=y, item_uuid=None)
                s_id += 1
        for y in range(5, 7):
            for x in range(6, GRID_WIDTH-5):
                if s_id in [i.shelf_id for i in self.items]:
                    item = next(i for i in self.items if i.shelf_id == s_id)
                    self.shelves[s_id] = Shelf(id=s_id, x=x, y=y, item_uuid=item.uuid)
                else:
                    self.shelves[s_id] = Shelf(id=s_id, x=x, y=y, item_uuid=None)
                s_id += 1
                
        # Create robots
        self.robots_physics = [
            RobotPhysics(0, 1, GRID_HEIGHT-1, self.shelves, self.stations),
            RobotPhysics(1, 2, GRID_HEIGHT-1, self.shelves, self.stations)
        ]
        
        self.algorithm = GroupRobotAlgorithm(GRID_WIDTH, GRID_HEIGHT, self.stations)
        self.tick = 0
        self.actions = [Direction.STAY for _ in self.robots_physics]

    def print_dashboard(self, robot_snapshots: List[RobotSnapshot], actions: List[Direction]):
        """Display current simulation state."""        
        print("\n" + "="*60)
        print(f" ⏱️  TICK: {self.tick}")
        print("-" * 60)

        # Build grid
        grid = [[' .  ' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        # Draw stations
        for s in self.stations:
            grid[s.y][s.x] = f'[S{s.id:<1}]'.ljust(4)

        # Draw shelves
        for s in self.shelves.values():
            if s.item_uuid is not None:
                grid[s.y][s.x] = f'X{s.id:<1}'.ljust(4)
            else:
                grid[s.y][s.x] = f'#{s.id:<1}'.ljust(4)

        # Draw robots (overlay)
        for r in robot_snapshots:
            symbol = f'{r.id:<1}R'.ljust(4)
            if r.carrying_item_uuid is not None:
                item = next((i for i in self.items if i.uuid == r.carrying_item_uuid), None)
                symbol = f'{r.id:<1}@{item.shelf_id}'.ljust(4)

            # Check for collisions
            cell_content = grid[r.y][r.x]
            if cell_content != ' .  ' and cell_content != '[S] ': 
                if 'R' in cell_content or '@' in cell_content:
                    grid[r.y][r.x] = ' 💥 '
                else:
                    grid[r.y][r.x] = symbol
            else:
                grid[r.y][r.x] = symbol

        # Print grid
        print("   " + "".join([f" {i:<2}".ljust(4) for i in range(GRID_WIDTH)]))
        for y in range(GRID_HEIGHT):
            row_str = "".join(grid[y])
            print(f"{y:<2} {row_str}")

        # Print item status
        print("-" * 60)
        print(f"Pending: {[i.shelf_id for i in self.items if i.status == ItemStatus.PENDING]}"
              + f" | In-progress: {[i.shelf_id for i in self.items if i.status == ItemStatus.IN_PROGRESS]}"
              + f" | Delivered: {[i.shelf_id for i in self.items if i.status == ItemStatus.COMPLETED]}")
        item_status = []
        for item in self.items:
            if item.status == ItemStatus.COMPLETED:
                status_symbol = "    ✅" 
            elif item.status == ItemStatus.IN_PROGRESS:
                status_symbol = "    🚚"
            else:
                status_symbol = "    ⌛"
            shelf_position = self.shelves[item.shelf_id]
            station_position = next(s for s in self.stations if s.id == item.target_station_id)
            status_str = ""
            status_str += f"{status_symbol} [{item.uuid}]: "
            status_str += f"Shelf {item.shelf_id} ({shelf_position.x}, {shelf_position.y}) ".ljust(17)
            status_str += f"-> Station {item.target_station_id} ({station_position.x}, {station_position.y})"
            item_status.append(status_str)
        item_status = item_status[:2] + ["    ..."]
        print(f"Item Status:")
        for status in item_status:
            print(status)
        
        # Print robot status
        robot_status = []
        for i, r in enumerate(robot_snapshots):
            if r.target is None:
                robot_status.append("    " + f"R{r.id}:({r.x},{r.y}) -> target:Idle".ljust(28) 
                                    + f"| carry: None".ljust(15) 
                                    + f"| action: {actions[i].name}")
            else:
                item = next((i for i in self.items if i.uuid == r.carrying_item_uuid), None)
                robot_status.append("    " + f"R{r.id}:({r.x},{r.y}) -> target:{r.target}".ljust(28) 
                                    + f"| carry: {item.shelf_id if item else 'None'}".ljust(15) 
                                    + f"| action: {actions[i].name}")
        print("Robots Status:")
        for status in robot_status:
            print(status)
        print("="*60)

    def parse_input(self) -> int:
        """
        Parse user input.
        Commands: [Enter] next frame | [t ShelfID StationID] add task | [q] quit
        """
        print("Instructions: [Enter] Next | [t ShelfID StationID] Add Task | [q] Quit")
        while True:
            raw = input("👉 Cmd: ").strip().lower()
            
            if raw == 'q':
                sys.exit(0)
            if raw == '':
                break
            
            if raw.startswith('t'):
                print(f"debug: Adding new task: {raw}")
                try:
                    cmds = raw.split()
                    if len(cmds) != 3:
                        print(f"  ❌ Invalid command format. Use: <t shelf_id station_id>, e.g.: t 1 0")
                        continue

                    shelf_id = int(cmds[1])
                    station_id = int(cmds[2])
                    if self.shelves[shelf_id] is None:
                        print(f"  ❌ Shelf ID {shelf_id} does not exist.")
                        continue
                    if self.shelves[shelf_id].item_uuid is not None:
                        print(f"  ❌ Shelf ID {shelf_id} already has an item.")
                        continue
                    if station_id not in [s.id for s in self.stations]:
                        print(f"  ❌ Station ID {station_id} does not exist.")
                        continue
                    else:
                        new_item = Item(shelf_id=shelf_id, uuid=shortuuid.ShortUUID().random(length=8), target_station_id=station_id)
                        self.items.append(new_item)
                        self.shelves[shelf_id].item_uuid = new_item.uuid
                        print(f"  ✅ Created new item [{new_item.uuid}]: Shelf {shelf_id} -> Station {station_id}")
                except:
                    print(f"  ❌ Failed to add item")

    def update_robot_targets(self):
        """Assign pending items to available robots."""
        pending_items = [item for item in self.items if item.status == ItemStatus.PENDING]
        if not pending_items:
            return

        for item in pending_items:
            # Skip if already assigned
            if any(robot.target == (self.shelves[item.shelf_id].x, self.shelves[item.shelf_id].y) for robot in self.robots_physics):
                continue
            
            # Assign to nearest available robot
            available_robots = [robot for robot in self.robots_physics if robot.target is None]
            if available_robots:
                nearest_robot = min(available_robots, key=lambda r: heuristic((r.x, r.y), (self.shelves[item.shelf_id].x, self.shelves[item.shelf_id].y)))
                nearest_robot.target = (self.shelves[item.shelf_id].x, self.shelves[item.shelf_id].y)
            else:
                break

    def update_map_status(self, actions: List[Direction]):
        """Update item pickups and deliveries."""
        for i, robot in enumerate(self.robots_physics):
            # Handle shelf pickups
            for shelf in self.shelves.values():
                if (shelf.x, shelf.y) == (robot.x, robot.y) and shelf.item_uuid is not None:
                    robot.carrying_item_uuid = shelf.item_uuid
                    shelf.item_uuid = None
                    item = next((it for it in self.items if it.uuid == robot.carrying_item_uuid), None)
                    item.status = ItemStatus.IN_PROGRESS
                    # Route to target station
                    for station in self.stations:
                        if station.id == item.target_station_id:
                            robot.target = (station.x, station.y)
                            break

            # Handle station deliveries
            for station in self.stations:
                if (station.x, station.y) == (robot.x, robot.y) and robot.carrying_item_uuid is not None:
                    item = next((it for it in self.items if it.uuid == robot.carrying_item_uuid), None)
                    if item:
                        item.status = ItemStatus.COMPLETED
                    robot.carrying_item_uuid = None
                    robot.target = None

    def run_interactive_simulation(self):
        """Main simulation loop."""
        while True:        
            # Get current robot state
            robot_snapshots = [r.get_snapshot() for r in self.robots_physics]

            # Display dashboard
            self.print_dashboard(robot_snapshots, self.actions)
            
            # Parse user input
            self.parse_input()
            
            # Update task assignments
            self.update_robot_targets()
            robot_snapshots = [r.get_snapshot() for r in self.robots_physics]
            
            # Compute actions
            self.actions = self.algorithm.decide_next_action(robot_snapshots, self.shelves)

            # Execute physics
            for i, r in enumerate(self.robots_physics):
                if i < len(self.actions):
                    r.tick(self.actions[i])

            # Update map state
            self.update_map_status(self.actions)
            self.tick += 1

"""
TODO:
- resolve robot-robot collisions
- path finding should consider other robots as dynamic obstacles
"""


if __name__ == "__main__":
    simulator = InteractiveSimulator()
    simulator.run_interactive_simulation()
