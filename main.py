import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

class Agent:
    """Base agent class"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        self.id = agent_id
        self.K = K  # Maximum bid value (bids from 1 to K)
        self.params = params or {}
        self.points = 0
        self.bid_history = []
        self.win_history = []
        self.total_reward = 0
        self.agent_type = self.__class__.__name__
        
    def choose_bid(self, round_num: int, game=None) -> int:
        """Choose a bid - to be overridden by subclasses"""
        return np.random.randint(1, self.K + 1)
    
    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        """Update agent's internal state after a round"""
        self.bid_history.append(all_bids[self.id])
        self.win_history.append(1 if winner_id == self.id else 0)
        if winner_id == self.id:
            self.total_reward += 1
        
    def reset(self):
        """Reset agent state for new simulation"""
        self.points = 0
        self.bid_history = []
        self.win_history = []
        self.total_reward = 0
    
    def get_avg_bid(self) -> float:
        """Get average bid value"""
        if self.bid_history:
            return np.mean(self.bid_history)
        return 0.0
    
    def get_win_rate(self) -> float:
        """Get win rate"""
        if self.win_history:
            return np.mean(self.win_history)
        return 0.0
    
    def get_agent_info(self) -> Dict:
        """Get agent information as dictionary"""
        return {
            'agent_id': self.id,
            'agent_type': self.agent_type,
            'K': self.K,
            'params': self.params
        }


class UniformRandomAgent(Agent):
    """Uniform Random Agent: chooses bids uniformly"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        
    def choose_bid(self, round_num: int, game=None) -> int:
        return np.random.randint(1, self.K + 1)


class PowerLawAgent(Agent):
    """Power-law / Low-Bias Agent: favors lower numbers"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.alpha = params.get('alpha', 1.0)
        self._update_distribution()
        
    def _update_distribution(self):
        """Update the probability distribution based on alpha"""
        k_values = np.arange(1, self.K + 1)
        probabilities = k_values ** (-self.alpha)
        self.prob_dist = probabilities / np.sum(probabilities)
        
    def choose_bid(self, round_num: int, game=None) -> int:
        return np.random.choice(np.arange(1, self.K + 1), p=self.prob_dist)

class ThresholdAgent(Agent):
    """Truncated Uniform (Threshold) Agent"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.T = params.get('T', max(1, K // 2))  # Threshold
        self.rho = params.get('rho', 0.7)  # Probability mass for low bids
        self._update_distribution()
        
    def _update_distribution(self):
        """Update the probability distribution"""
        self.prob_dist = np.zeros(self.K)
        for k in range(1, self.K + 1):
            if k <= self.T:
                self.prob_dist[k-1] = self.rho / self.T
            else:
                self.prob_dist[k-1] = (1 - self.rho) / max(1, (self.K - self.T))
        self.prob_dist = self.prob_dist / np.sum(self.prob_dist)  # Normalize
        
    def choose_bid(self, round_num: int, game=None) -> int:
        return np.random.choice(np.arange(1, self.K + 1), p=self.prob_dist)


class EmpiricalFrequencyAgent(Agent):
    """Empirical Frequency Estimator Agent"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.beta = params.get('beta', 1.0)  # Rationality parameter
        self.empirical_freq = np.ones(K) / K  # Initial belief
        
    def choose_bid(self, round_num: int, game=None) -> int:
        if round_num == 0 or game is None or not hasattr(game, 'agents'):
            # First round or no game data: choose uniformly
            return np.random.randint(1, self.K + 1)
        
        # Get history from game
        history = game.history if hasattr(game, 'history') else None
        
        if history is None:
            return np.random.randint(1, self.K + 1)
        
        # Update empirical frequency based on history
        N = len(game.agents)
        total_observations = (N - 1) * round_num
        
        if total_observations > 0 and 'bid_history' in history:
            # Count bids from other agents
            freq_counts = np.zeros(self.K)
            
            # Check if we can access agent bid histories through game
            if hasattr(game, 'agents'):
                for agent in game.agents:
                    if agent.id != self.id and hasattr(agent, 'bid_history'):
                        for bid in agent.bid_history[:round_num]:
                            if 1 <= bid <= self.K:
                                freq_counts[bid-1] += 1
            # Alternative: try to get from history dictionary
            elif 'bid_history' in history:
                for agent_id, bids in history['bid_history'].items():
                    if agent_id != self.id:
                        for bid in bids[:round_num]:
                            if 1 <= bid <= self.K:
                                freq_counts[bid-1] += 1
            
            if np.sum(freq_counts) > 0:
                self.empirical_freq = freq_counts / total_observations
                
                # Avoid zero probabilities
                self.empirical_freq = np.clip(self.empirical_freq, 1e-6, 1)
                self.empirical_freq = self.empirical_freq / np.sum(self.empirical_freq)
        
        # Calculate probability of uniqueness for each bid
        N_others = N - 1
        P_unique = np.zeros(self.K)
        
        for k in range(self.K):
            fk = self.empirical_freq[k]
            # Probability that k is unique among N-1 others
            if N_others >= 1:
                P_unique[k] = N_others * fk * ((1 - fk) ** (N_others - 1))
        
        # Calculate utility
        k_values = np.arange(1, self.K + 1).astype(float)
        U = P_unique / k_values
        
        # Avoid NaN/Inf
        U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Softmax policy
        if np.sum(U) > 0 and not np.all(U == 0):
            exp_U = np.exp(self.beta * U - np.max(self.beta * U))  # For numerical stability
            prob_dist = exp_U / np.sum(exp_U)
        else:
            prob_dist = np.ones(self.K) / self.K
            
        # Choose bid based on probability distribution
        return np.random.choice(np.arange(1, self.K + 1), p=prob_dist)
    
    def reset(self):
        super().reset()
        self.empirical_freq = np.ones(self.K) / self.K  # Reset belief


class BestResponseAgent(Agent):
    """Best-Response-to-Empirical Distribution Agent"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.beta = params.get('beta', 1.0)  # Rationality parameter
        self.empirical_dist = np.ones(K) / K  # Initial belief about others
        
    def choose_bid(self, round_num: int, game=None) -> int:
        if round_num == 0 or game is None:
            # First round: choose uniformly
            return np.random.randint(1, self.K + 1)
        
        # Get history from game
        history = game.history if hasattr(game, 'history') else None
        
        if history is None:
            return np.random.randint(1, self.K + 1)
        
        # Update empirical distribution
        N = len(game.agents)
        total_observations = (N - 1) * round_num
        
        if total_observations > 0:
            # Count bids from other agents
            dist_counts = np.zeros(self.K)
            
            # Check if we can access agent bid histories through game
            if hasattr(game, 'agents'):
                for agent in game.agents:
                    if agent.id != self.id and hasattr(agent, 'bid_history'):
                        for bid in agent.bid_history[:round_num]:
                            if 1 <= bid <= self.K:
                                dist_counts[bid-1] += 1
            # Alternative: try to get from history dictionary
            elif 'bid_history' in history:
                for agent_id, bids in history['bid_history'].items():
                    if agent_id != self.id:
                        for bid in bids[:round_num]:
                            if 1 <= bid <= self.K:
                                dist_counts[bid-1] += 1
            
            if np.sum(dist_counts) > 0:
                self.empirical_dist = dist_counts / total_observations
                
                # Avoid zero probabilities
                self.empirical_dist = np.clip(self.empirical_dist, 1e-6, 1)
                self.empirical_dist = self.empirical_dist / np.sum(self.empirical_dist)
        
        # Calculate probability of winning for each bid
        N_others = N - 1
        P_win = np.zeros(self.K)
        
        for k in range(self.K):
            # Probability no one else bids k
            P_no_k = (1 - self.empirical_dist[k]) ** N_others
            
            # Probability no lower unique bid exists
            P_no_lower_unique = 1.0
            for j in range(k):  # j < k
                # Probability that j is a unique bid among others
                if N_others >= 1:
                    P_j_unique = N_others * self.empirical_dist[j] * ((1 - self.empirical_dist[j]) ** (N_others - 1))
                    # Probability j is NOT a unique bid
                    P_j_not_unique = 1 - P_j_unique
                    P_no_lower_unique *= P_j_not_unique
            
            P_win[k] = P_no_k * P_no_lower_unique
        
        # Avoid NaN/Inf
        P_win = np.nan_to_num(P_win, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Softmax policy
        if np.sum(P_win) > 0 and not np.all(P_win == 0):
            exp_P = np.exp(self.beta * P_win - np.max(self.beta * P_win))
            prob_dist = exp_P / np.sum(exp_P)
        else:
            prob_dist = np.ones(self.K) / self.K
            
        return np.random.choice(np.arange(1, self.K + 1), p=prob_dist)
    
    def reset(self):
        super().reset()
        self.empirical_dist = np.ones(self.K) / self.K

class ReinforcementLearningAgent(Agent):
    """Reinforcement Learning (Bandit) Agent"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.beta = params.get('beta', 1.0)  # Rationality parameter
        self.eta = params.get('eta', 0.1)    # Learning rate
        self.Q_values = np.zeros(K)          # Action values
        self.action_counts = np.zeros(K)     # Count of times each action taken
        
    def choose_bid(self, round_num: int, game=None) -> int:
        # Softmax policy based on Q-values
        if round_num == 0 or np.sum(self.Q_values) == 0:
            # First round: choose uniformly
            return np.random.randint(1, self.K + 1)
        
        # Apply softmax to Q-values
        exp_Q = np.exp(self.beta * self.Q_values - np.max(self.beta * self.Q_values))
        prob_dist = exp_Q / np.sum(exp_Q)
        
        return np.random.choice(np.arange(1, self.K + 1), p=prob_dist)
    
    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        super().update(round_num, all_bids, winner_id, game)
        
        # Update Q-value for the chosen action
        my_bid = all_bids[self.id]
        reward = 1 if winner_id == self.id else 0
        
        # Update action count
        self.action_counts[my_bid-1] += 1
        
        # Update Q-value using learning rate
        self.Q_values[my_bid-1] += self.eta * (reward - self.Q_values[my_bid-1])
    
    def reset(self):
        super().reset()
        self.Q_values = np.zeros(self.K)
        self.action_counts = np.zeros(self.K)


class ReplicatorDynamicsAgent(Agent):
    """Replicator Dynamics Agent (Population-Level)"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.population_dist = np.ones(K) / K  # Initial population distribution
        self.fitness_history = []
        
    def choose_bid(self, round_num: int, game=None) -> int:
        # Choose bid according to current population distribution
        return np.random.choice(np.arange(1, self.K + 1), p=self.population_dist)
    
    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        super().update(round_num, all_bids, winner_id, game)
        
        # Get history from game if needed
        history = game.history if game is not None and hasattr(game, 'history') else None
        
        if round_num > 0:
            # Calculate fitness for each bid
            fitness = np.zeros(self.K)
            
            # Count wins for each bid value
            if winner_id is not None:
                winning_bid = all_bids[winner_id]
                fitness[winning_bid-1] += 1
            
            # Average fitness
            avg_fitness = np.sum(self.population_dist * fitness)
            
            # Update population distribution
            if avg_fitness > 0:
                self.population_dist = self.population_dist * fitness / avg_fitness
                # Normalize
                self.population_dist = self.population_dist / np.sum(self.population_dist)
            
            self.fitness_history.append(fitness.copy())
    
    def reset(self):
        super().reset()
        self.population_dist = np.ones(self.K) / self.K
        self.fitness_history = []


class LevelKAgent(Agent):
    """Level-k / Cognitive Hierarchy Agent"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.level = params.get('level', 1)  # Cognitive level
        self.beta = params.get('beta', 1.0)  # Rationality parameter
        self.level_dist = params.get('level_dist', [0.5, 0.3, 0.2])  # Distribution of levels in population
        self.level_strategies = []
        
        # Generate strategies for each level up to self.level
        self._generate_level_strategies()
        
    def _generate_level_strategies(self):
        """Generate strategies for each cognitive level"""
        self.level_strategies = []
        
        # Level-0: uniform random
        level0_dist = np.ones(self.K) / self.K
        self.level_strategies.append(level0_dist)
        
        # Higher levels: best response to mixture of lower levels
        for L in range(1, self.level + 1):
            # Create mixture of lower level strategies
            mixture_dist = np.zeros(self.K)
            for l in range(L):
                if l < len(self.level_dist):
                    weight = self.level_dist[l] / sum(self.level_dist[:L])
                    mixture_dist += weight * self.level_strategies[l]
            
            # Best response (softmax) to mixture
            # Simplified: assume others play according to mixture_dist
            N_others = 5  # Approximate number of other agents
            P_win = np.zeros(self.K)
            
            for k in range(self.K):
                # Probability no one else bids k
                P_no_k = (1 - mixture_dist[k]) ** N_others
                
                # Probability no lower unique bid exists
                P_no_lower_unique = 1.0
                for j in range(k):
                    if N_others >= 1:
                        P_j_unique = N_others * mixture_dist[j] * ((1 - mixture_dist[j]) ** (N_others - 1))
                        P_no_lower_unique *= (1 - P_j_unique)
                
                P_win[k] = P_no_k * P_no_lower_unique
            
            # Softmax
            if np.sum(P_win) > 0:
                exp_P = np.exp(self.beta * P_win - np.max(self.beta * P_win))
                levelL_dist = exp_P / np.sum(exp_P)
            else:
                levelL_dist = np.ones(self.K) / self.K
                
            self.level_strategies.append(levelL_dist)
    
    def choose_bid(self, round_num: int, game=None) -> int:
        # Use the strategy for this agent's level
        if self.level < len(self.level_strategies):
            dist = self.level_strategies[self.level]
        else:
            dist = np.ones(self.K) / self.K
            
        return np.random.choice(np.arange(1, self.K + 1), p=dist)


class MutationNoiseAgent(Agent):
    """Agent with mutation / noise"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.base_agent_type = params.get('base_agent_type', 'uniform')
        self.mu = params.get('mu', 0.1)  # Mutation rate
        
        # Create the base agent
        base_params = params.get('base_params', {})
        if self.base_agent_type == 'uniform':
            self.base_agent = UniformRandomAgent(agent_id, K, base_params)
        elif self.base_agent_type == 'powerlaw':
            self.base_agent = PowerLawAgent(agent_id, K, base_params)
        elif self.base_agent_type == 'threshold':
            self.base_agent = ThresholdAgent(agent_id, K, base_params)
        elif self.base_agent_type == 'empirical':
            self.base_agent = EmpiricalFrequencyAgent(agent_id, K, base_params)
        elif self.base_agent_type == 'bestresponse':
            self.base_agent = BestResponseAgent(agent_id, K, base_params)
        elif self.base_agent_type == 'rl':
            self.base_agent = ReinforcementLearningAgent(agent_id, K, base_params)
        elif self.base_agent_type == 'replicator':
            self.base_agent = ReplicatorDynamicsAgent(agent_id, K, base_params)
        elif self.base_agent_type == 'levelk':
            self.base_agent = LevelKAgent(agent_id, K, base_params)
        else:
            self.base_agent = UniformRandomAgent(agent_id, K, base_params)
    
    def choose_bid(self, round_num: int, game=None) -> int:
        # Get base agent's bid
        base_bid = self.base_agent.choose_bid(round_num, game)
        
        # Apply mutation with probability mu
        if np.random.random() < self.mu:
            # Mutate: choose uniformly from all bids
            return np.random.randint(1, self.K + 1)
        else:
            return base_bid
    
    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        super().update(round_num, all_bids, winner_id, game)
        # Also update the base agent
        self.base_agent.update(round_num, all_bids, winner_id, game)
    
    def reset(self):
        super().reset()
        self.base_agent.reset()


class LowestUniqueBidGame:
    """Main simulation class for the lowest unique bid game with dataset recording"""
    
    def __init__(self, K: int = 10, agents: Optional[List[Agent]] = None, 
                 verbose: bool = False, record_dataset: bool = True):
        """
        Initialize the game
        
        Parameters:
        - K: Maximum bid value (bids are 1 to K)
        - agents: List of agent objects
        - verbose: Whether to print detailed information
        - record_dataset: Whether to record bid dataset
        """
        self.K = K
        self.agents = agents if agents else []
        self.verbose = verbose
        self.record_dataset = record_dataset
        self.round_num = 0
        
        # History tracking - using your expected structure
        self.history = {
            'agents': {agent.id: agent for agent in self.agents},
            'bid_history': defaultdict(list),
            'winners': [],
            'all_bids': []
        }
        
        # Dataset storage for bids and outcomes
        self.bid_dataset = []  # List of dictionaries, each representing a round
        self.agent_map = {agent.id: agent for agent in self.agents}
        
        # Statistics
        self.bid_counts = np.zeros(K)
        self.win_counts = np.zeros(K)
        
        # Metadata
        self.game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metadata = {
            'game_id': self.game_id,
            'K': K,
            'num_agents': len(self.agents),
            'agent_types': {},
            'created_at': datetime.now().isoformat()
        }
        
        # Record agent types in metadata
        agent_type_counts = {}
        for agent in self.agents:
            agent_type = type(agent).__name__
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1
        self.metadata['agent_types'] = agent_type_counts
    
    def add_agent(self, agent: Agent):
        """Add an agent to the game"""
        self.agents.append(agent)
        self.agent_map[agent.id] = agent
        self.history['agents'][agent.id] = agent
        self.metadata['num_agents'] = len(self.agents)
        
        # Update agent type count
        agent_type = type(agent).__name__
        self.metadata['agent_types'][agent_type] = self.metadata['agent_types'].get(agent_type, 0) + 1
    
    def determine_winner(self, bids: Dict[int, int]) -> Tuple[Optional[int], Optional[int]]:
        """
        Determine the lowest unique bid winner
        
        Returns:
        - winner_id: ID of winning agent (None if no winner)
        - winning_bid: The winning bid value (None if no winner)
        """
        # Count frequency of each bid
        freq = Counter(bids.values())
        
        # Find unique bids
        unique_bids = [bid for bid, count in freq.items() if count == 1]
        
        if not unique_bids:
            return None, None  # No unique bid
        
        # Find the lowest unique bid
        lowest_unique = min(unique_bids)
        
        # Find the agent(s) with the lowest unique bid
        winners = [agent_id for agent_id, bid in bids.items() if bid == lowest_unique]
        
        # Should be exactly one winner since it's unique
        if winners:
            return winners[0], lowest_unique
        return None, None
    
    def play_round(self) -> Dict:
        """
        Play one round of the game and record data
        
        Returns:
        - Dictionary with round results including dataset record
        """
        bids = {}
        
        # Each agent chooses a bid - pass self (the game) as history
        for agent in self.agents:
            bid = agent.choose_bid(self.round_num, self)
            bids[agent.id] = bid
        
        # Determine winner
        winner_id, winning_bid = self.determine_winner(bids)
        
        # Update scores and agent states
        if winner_id is not None:
            winning_agent = self.agent_map[winner_id]
            winning_agent.points += 1
            
            # Update statistics
            self.win_counts[winning_bid-1] += 1
        
        # Update all agents - pass self as game
        for agent in self.agents:
            agent.update(self.round_num, bids, winner_id, self)
        
        # Update bid counts and history
        for bid in bids.values():
            self.bid_counts[bid-1] += 1
        
        # Update history dictionary
        for agent_id, bid in bids.items():
            self.history['bid_history'][agent_id].append(bid)
        
        self.history['winners'].append(winner_id)
        self.history['all_bids'].append(bids.copy())
        
        # Create detailed round record for dataset
        round_record = self._create_round_record(bids, winner_id, winning_bid)
        
        # Add to bid dataset if recording is enabled
        if self.record_dataset:
            self.bid_dataset.append(round_record)
        
        # Print round info if verbose
        if self.verbose:
            print(f"\nRound {self.round_num + 1}:")
            print(f"  Bids: {bids}")
            if winner_id is not None:
                print(f"  Winner: Agent {winner_id} with bid {winning_bid}")
                print(f"  Winner's Price: {winning_bid}")
            else:
                print("  No winner (no unique bid)")
        
        self.round_num += 1
        
        return round_record
    
    def _create_round_record(self, bids: Dict[int, int], winner_id: Optional[int], 
                            winning_bid: Optional[int]) -> Dict:
        """
        Create a detailed record of a round for the dataset
        
        Parameters:
        - bids: Dictionary mapping agent_id -> bid
        - winner_id: ID of winning agent (None if no winner)
        - winning_bid: The winning bid value (None if no winner)
        
        Returns:
        - Dictionary with detailed round information
        """
        # Count bid frequencies
        bid_frequencies = Counter(bids.values())
        
        # Create record
        record = {
            'game_id': self.game_id,
            'round': self.round_num,
            'timestamp': datetime.now().isoformat(),
            'num_agents': len(self.agents),
            'bids': bids.copy(),  # Raw bids
            'bid_frequencies': dict(bid_frequencies),  # Count of each bid value
            'unique_bids': [bid for bid, count in bid_frequencies.items() if count == 1],
            'has_winner': winner_id is not None,
            'winner_id': winner_id,
            'winner_bid': winning_bid,
            'winner_agent_type': None,
            'agent_bids': []  # Detailed bid information per agent
        }
        
        # Add detailed agent information
        for agent_id, bid in bids.items():
            agent = self.agent_map[agent_id]
            agent_info = {
                'agent_id': agent_id,
                'agent_type': type(agent).__name__,
                'bid': bid,
                'is_winner': agent_id == winner_id,
                'agent_params': agent.params
            }
            record['agent_bids'].append(agent_info)
            
            # Update winner's agent type
            if agent_id == winner_id:
                record['winner_agent_type'] = type(agent).__name__
        
        return record
    
    def play_multiple_rounds(self, num_rounds: int) -> List[Dict]:
        """Play multiple rounds of the game and collect dataset"""
        round_results = []
        for i in range(num_rounds):
            result = self.play_round()
            round_results.append(result)
            
            # Print progress for long simulations
            if self.verbose and (i+1) % 100 == 0:
                print(f"Completed {i+1}/{num_rounds} rounds")
        
        return round_results
    
    # [Rest of the methods remain the same as in the previous implementation]
    # get_bid_dataset, get_winning_prices, export_dataset, get_agent_stats, 
    # get_bid_distribution, get_game_summary, reset, etc.


# Helper function to create agents
def create_agent_pool(K: int = 10, num_agents: int = 20) -> List[Agent]:
    """Create a diverse pool of agents for simulation"""
    agents = []
    
    # Agent configurations
    agent_configs = [
        # Uniform Random Agents
        ('uniform', 3, {}),
        
        # Power Law Agents with different alpha values
        ('powerlaw', 2, {'alpha': 0.5}),
        ('powerlaw', 2, {'alpha': 1.0}),
        ('powerlaw', 2, {'alpha': 2.0}),
        
        # Threshold Agents
        ('threshold', 2, {'T': max(1, K//3), 'rho': 0.8}),
        ('threshold', 2, {'T': max(1, 2*K//3), 'rho': 0.5}),
        
        # Empirical Frequency Agents
        ('empirical', 2, {'beta': 0.5}),
        ('empirical', 1, {'beta': 2.0}),
        
        # Best Response Agents
        ('bestresponse', 2, {'beta': 1.0}),
        
        # Reinforcement Learning Agents
        ('rl', 2, {'beta': 1.0, 'eta': 0.1}),
        
        # Replicator Dynamics Agents
        ('replicator', 2, {}),
        
        # Level-K Agents
        ('levelk', 1, {'level': 1, 'beta': 1.0}),
        ('levelk', 1, {'level': 2, 'beta': 1.0}),
        
        # Mutation Agents
        ('mutation', 1, {'base_agent_type': 'empirical', 'mu': 0.1}),
    ]
    
    agent_id = 0
    for agent_type, count, params in agent_configs:
        for _ in range(count):
            if agent_type == 'uniform':
                agents.append(UniformRandomAgent(agent_id, K, params))
            elif agent_type == 'powerlaw':
                agents.append(PowerLawAgent(agent_id, K, params))
            elif agent_type == 'threshold':
                agents.append(ThresholdAgent(agent_id, K, params))
            elif agent_type == 'empirical':
                agents.append(EmpiricalFrequencyAgent(agent_id, K, params))
            elif agent_type == 'bestresponse':
                agents.append(BestResponseAgent(agent_id, K, params))
            elif agent_type == 'rl':
                agents.append(ReinforcementLearningAgent(agent_id, K, params))
            elif agent_type == 'replicator':
                agents.append(ReplicatorDynamicsAgent(agent_id, K, params))
            elif agent_type == 'levelk':
                agents.append(LevelKAgent(agent_id, K, params))
            elif agent_type == 'mutation':
                agents.append(MutationNoiseAgent(agent_id, K, params))
            
            agent_id += 1
    
    return agents


# Main execution example
if __name__ == "__main__":
    # Create agents
    K = 30
    agents = create_agent_pool(K, 20)
    
    # Create game
    game = LowestUniqueBidGame(K, agents, verbose=True, record_dataset=True)
    
    # Play some rounds
    print("Starting simulation...")
    game.play_multiple_rounds(10)
    
    # Print results
    print(f"\nTotal rounds played: {game.round_num}")
    print(f"Winning prices: {[record['winner_bid'] for record in game.bid_dataset if record['has_winner']]}")