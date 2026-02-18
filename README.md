# Agent based unique bid simulation
**A Multi-Agent Simulation Framework for the Lowest Unique Bid Game**

## Abstract
This repository presents a simulation framework for studying the **Lowest Unique Bid (LUB) game** in a multi-agent setting. The framework supports heterogeneous agents with distinct strategic, learning-based, and evolutionary behaviors. The primary objective of the project is to analyze strategic interactions, learning dynamics, and emergent outcomes when diverse agent populations repeatedly participate in a lowest unique bid auction.

---

## 1. Problem Definition
In the Lowest Unique Bid game, a fixed number of agents simultaneously submit integer bids from a bounded range \([1, K]\). A bid is considered *unique* if it is submitted by exactly one agent. The agent who submits the **lowest unique bid** wins the round. If no bid is unique, the round results in no winner.

This game is of interest in behavioral economics, auction theory, and multi-agent systems, as it exhibits coordination failures, strategic crowding, and non-trivial equilibria.

---

## 2. Agent Models
The simulation includes multiple agent classes representing different behavioral assumptions:

- **UniformRandomAgent**: Uniform random bidding over the full bid range.
- **PowerLawAgent**: Bidding with a power-law bias toward lower values.
- **ThresholdAgent**: Truncated uniform strategy concentrating probability mass below a threshold.
- **EmpiricalFrequencyAgent**: Adaptive strategy based on observed bid frequencies of other agents.
- **BestResponseAgent**: Approximate best-response to an empirical distribution of opponent bids.
- **ReinforcementLearningAgent**: Bandit-style reinforcement learning with softmax action selection.
- **ReplicatorDynamicsAgent**: Evolutionary dynamics based on population-level fitness.
- **LevelKAgent**: Cognitive hierarchy (level-k) reasoning model.
- **MutationNoiseAgent**: Strategy perturbation via stochastic mutation applied to a base agent.

Each agent updates its internal state after every round based on observed outcomes.

---

## 3. Simulation Framework
The core simulation is implemented in the `LowestUniqueBidGame` class. For each round:

1. All agents independently select a bid.
2. The lowest unique bid is identified.
3. The winning agent (if any) is recorded.
4. Agents update their internal states.
5. Round-level and agent-level data are logged.

The simulation supports repeated play over an arbitrary number of rounds and records a complete dataset suitable for post-hoc statistical analysis.

---

## 4. Output Data
All outputs are exported in **CSV format** for maximal interoperability and reproducibility. Each simulation run generates a timestamped directory under `outputs/` containing:

### Core Datasets
- **rounds.csv**  
  One row per round, including winner information, winning bid, and number of unique bids.

- **agent_rounds.csv**  
  One row per (agent, round) pair, including bid value, win indicator, and agent type.

- **summary.csv**  
  A single-row summary of global statistics (e.g., winner rate, average winning bid).

- **agent_type_counts.csv**  
  Counts of agents per strategy type.

- **metadata.csv**  
  Key–value metadata describing the simulation configuration.

### Figures
A set of high-resolution figures is saved under `outputs/.../figures/`, including:
- Distribution of winning bids
- Winner rate over time (rolling average)
- Win rate by agent type
- Average bid by agent type
- Bid distributions by agent type
- Cumulative wins of top-performing agents

---

## 5. Reproducibility
- A fixed random seed is used by default to ensure reproducible results.
- All simulation parameters and agent compositions are recorded in the exported metadata.
- Output files are deterministic given the same seed and configuration.

---

## 6. Usage

### Dependencies
Install required dependencies using:
```bash
pip install -r requirements.txt
```
Execution
Run the simulation with:
```bash
python main.py
```
Simulation parameters (e.g., number of agents, bid range, number of rounds) can be modified in the main() function.



## 7. Intended Use
This framework is designed for:

Research on auction dynamics and strategic interaction

Comparative evaluation of learning and heuristic agent strategies

Educational demonstrations of multi-agent systems

Generation of structured datasets for further statistical analysis

## 8. Notes

The outputs/ directory is excluded from version control.

The project is self-contained and does not rely on external datasets.

