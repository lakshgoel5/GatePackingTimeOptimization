# Gate Packing Time Optimization

A VLSI circuit optimization tool that minimizes critical path delay by efficiently placing logic gates while considering both area constraints and signal propagation delays.

## Project Description

This project implements a sophisticated algorithm for optimizing the placement of logic gates in VLSI circuits. The primary goal is to minimize the critical path delay (the longest path that signals must travel), which directly impacts the circuit's maximum operating frequency.

The optimization process uses simulated annealing to find efficient placements while considering:
- Gate dimensions and areas
- Wire lengths and associated delays
- Critical path analysis
- Area optimization with efficient space utilization
- Circuit topology and connectivity

## Data Structures Used

1. **Dictionaries**: Used extensively to store gate properties, placements, wire connections, and pin structures
2. **Sets**: Used for storing connections between pins and tracking visited nodes
3. **Lists**: Used for managing available spaces, sorting gates by size, and maintaining critical paths
4. **Custom Exception Class**: For detecting cycles in the circuit graph

## Algorithm Flow

The optimization process consists of several key steps:

## Algorithm Flow (Detailed)

### 1. Initial Parsing and Setup

The algorithm begins by reading and parsing the circuit description from an input file:
- **Gate Information**: Dimensions (width, height) and intrinsic delays of each logic gate
- **Wire Connections**: Source and destination pins between gates
- **Pin Configurations**: Locations of input/output pins on gates
- **Primary Inputs**: Identification of external input signals to the circuit

During this phase, data structures are populated to represent the circuit topology:
- Gate dictionaries storing physical and electrical properties
- Wire connectivity maps between pins
- Pin group mappings for related signals

### 2. Area Optimization

This phase focuses on efficient placement of gates to minimize the total area:

- **Gate Sorting**: Gates are sorted by size in descending order (largest first) to optimize space usage
- **Space Management**:
  - The algorithm maintains a list of available spaces, sorted by area
  - When placing a gate, it searches for the best-fit space to minimize wasted area
  - After placing a gate, the remaining space is split into two new spaces and reinserted into the sorted list
  
- **Placement Strategies**:
  - For the first gate, placement starts at the origin (0,0)
  - Subsequent gates are placed using either:
    - The "best-fit" approach in available spaces
    - Addition to the right or above the current bounding box when necessary
  - The algorithm dynamically decides between horizontal or vertical expansion based on which creates less wasted space

### 3. Delay Calculation and Critical Path Analysis

This phase computes signal propagation delays and identifies the performance-limiting path:

- **Wire Length Calculation**:
  - Manhattan distance is used to calculate wire lengths between connected gates
  - Wire length × delay factor gives the signal delay between gates
  
- **Path Traversal**:
  - A dynamic programming approach computes the maximum delay from any input to any output
  - The algorithm recursively explores all paths through the circuit
  - Memoization prevents recalculating delays for shared sub-paths
  
- **Critical Path Identification**:
  - The algorithm identifies the sequence of gates and wires forming the longest delay path
  - This path determines the maximum operating frequency of the circuit
  - The path is stored for both reporting and targeted optimization

### 4. Multi-Stage Simulated Annealing Optimization

The optimization employs a three-stage simulated annealing approach to minimize the critical path delay:

- **Initial Temperature Selection**:
  - High initial temperature allows exploration of diverse solutions
  - Temperature gradually decreases (cooling rate α = 0.95) to focus on improvements
  
- **Stage 1: Cluster-based Optimization**:
  - Identifies gate clusters with high connectivity
  - Places connected gates closer together to minimize wire lengths
  - Uses a probabilistic acceptance criterion to occasionally accept worse solutions
  
- **Stage 2: Global Optimization**:
  - Multiple iterations of simulated annealing with different starting points
  - Gates are moved considering the entire circuit
  - Each iteration saves the best solution found
  
- **Stage 3: Fine-tuning**:
  - Focused optimization targeting gates on the critical path
  - Small incremental moves (±1 position) to find local optimality
  - Progressive narrowing of accepted solutions as temperature decreases
  
- **Move Validation**:
  - Each proposed move is checked for overlap with other gates
  - The algorithm calculates the new delay after each move
  - Moves that reduce delay are always accepted
  - Moves that increase delay are accepted with probability exp(-ΔCost/T)

### 5. Output Generation and Processing

The final phase prepares and writes the optimization results:

- **Coordinate Normalization**:
  - Gate coordinates are normalized to start at (0,0)
  - This ensures efficient use of the coordinate space
  
- **Bounding Box Calculation**:
  - Determines the minimum rectangle that contains all gates
  - Calculates the total area required by the placement
  
- **Result Output**:
  - Writes the bounding box dimensions
  - Lists the gates in the critical path
  - Reports the critical path delay
  - Records the optimized (x,y) coordinates for each gate placement

5. **Output Generation**:
   - Write optimized gate placements to output file
   - Report bounding box dimensions
   - List the critical path and its delay

## How to Run

1. Prepare an input file (`input.txt`) with the following format:
   ```
   [Number of gates]
   [Gate name] [Width] [Height] [Delay]
   ...
   [Number of wires]
   [Source pin] [Destination pin]
   ...
   ```

2. Run the program:
   ```powershell
   python main.py
   ```

3. The program will generate an `output.txt` file with:
   - The bounding box dimensions
   - The critical path through the circuit
   - The critical path delay
   - Optimized gate placements

## Key Features

- **Multi-level Optimization**: Combines area optimization with delay minimization
- **Adaptive Placement**: Adjusts gate positions based on connectivity
- **Efficient Space Management**: Intelligent allocation of free spaces
- **Temperature-based Acceptance**: Gradually focuses on improvements as optimization proceeds
- **Critical Path Analysis**: Identifies and optimizes the performance-limiting path

## Performance Considerations

- The execution time is displayed after running the program
- For large circuits, multiple optimization iterations help find better solutions
- The algorithm balances area efficiency with delay minimization