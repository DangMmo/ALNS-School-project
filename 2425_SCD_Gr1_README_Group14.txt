Main Components
The Python implementation consists of several key modules organized hierarchically:

Core Data Structures:

Node, Hub, Satellite, and Customer classes define the problem entities

SERoute and FERoute classes handle second-echelon and first-echelon vehicle routing respectively

SolutionState class manages the overall solution representation for ALNS

Algorithm Implementation:

build_initial_solution() function generates starting solutions using greedy heuristics

ALNS framework with destroy operators (random customer removal) and repair operators (greedy insertion)

Simulated Annealing acceptance criterion for solution evaluation

Navigation Structure
Problem Setup (Lines 1-200):

Constants and configuration parameters

Helper functions for distance calculations and travel time computations

Basic data structure definitions

Solution Management (Lines 200-800):

Route calculation and feasibility checking methods

Synchronization logic between first and second echelons

Cost computation and constraint validation

ALNS Implementation (Lines 800-1200):

Destroy and repair operator implementations

Adaptive learning mechanisms and operator selection

Main ALNS iteration loop with acceptance criteria

Execution Section (Lines 1200+):

Input file parsing and data preprocessing

Algorithm execution with parameter configuration

Results output and visualization generation

FINAL-REPORT.pdf - Academic Documentation
Document Structure
The 52-page report follows a standard academic format with six main chapters:

Chapter 1: Introduction (Pages 8-12)

Background on urban logistics and e-commerce challenges

Problem statement defining the 2E-VRP-PDD variant

Study objectives and scope limitations

Project planning with Gantt chart visualization

Chapter 2: Literature Review (Pages 13-16)

Two-echelon systems with integrated pickup and delivery operations

Time-related constraints and synchronization requirements

Solution methodologies for complex routing problems

Research gap identification and contribution positioning

Chapter 3: Methodology (Pages 17-20)

Solution approach justification and comparison

Metaheuristics evaluation covering Genetic Algorithms, Variable Neighborhood Search, and ALNS

Rationale for ALNS selection over alternative methods

Navigation Guide
Technical Content (Chapters 4-5):

Chapter 4 details the solution development including problem formulation, data structures, and ALNS implementation

Chapter 5 presents experimental results with data collection, processing, and sensitivity analysis

Supporting Materials:

Appendices contain code repository links, data files, and complete results

Figures and tables are numbered sequentially (e.g., Figure 4.1, Table 5.1)

References section includes 24 academic sources with DOI links

Key Findings Access:

Performance results are summarized in Tables 5.1-5.3 showing 12-16% average improvements

Sensitivity analysis results appear in Tables 5.4-5.5 with parameter tuning insights

Visual route solutions and performance graphs are integrated throughout Chapter 5

The documents complement each other, with the Python file providing the executable implementation and the PDF offering comprehensive theoretical background, experimental validation, and academic context for the 2E-VRP-PDD problem and ALNS solution approach.