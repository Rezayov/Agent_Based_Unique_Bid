import os
import json
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Agents
# =========================
class Agent:
    """Base agent class"""
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        self.id = agent_id
        self.K = K
        self.params = params or {}
        self.points = 0
        self.bid_history = []
        self.win_history = []
        self.total_reward = 0
        self.agent_type = self.__class__.__name__

    def choose_bid(self, round_num: int, game=None) -> int:
        return np.random.randint(1, self.K + 1)

    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        self.bid_history.append(all_bids[self.id])
        self.win_history.append(1 if winner_id == self.id else 0)
        if winner_id == self.id:
            self.total_reward += 1

    def reset(self):
        self.points = 0
        self.bid_history = []
        self.win_history = []
        self.total_reward = 0

    def get_avg_bid(self) -> float:
        return float(np.mean(self.bid_history)) if self.bid_history else 0.0

    def get_win_rate(self) -> float:
        return float(np.mean(self.win_history)) if self.win_history else 0.0


class UniformRandomAgent(Agent):
    def choose_bid(self, round_num: int, game=None) -> int:
        return np.random.randint(1, self.K + 1)


class PowerLawAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.alpha = params.get('alpha', 1.0)
        self._update_distribution()

    def _update_distribution(self):
        k_values = np.arange(1, self.K + 1)
        probabilities = k_values ** (-self.alpha)
        self.prob_dist = probabilities / np.sum(probabilities)

    def choose_bid(self, round_num: int, game=None) -> int:
        return int(np.random.choice(np.arange(1, self.K + 1), p=self.prob_dist))


class ThresholdAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.T = params.get('T', max(1, K // 2))
        self.rho = params.get('rho', 0.7)
        self._update_distribution()

    def _update_distribution(self):
        self.prob_dist = np.zeros(self.K)
        for k in range(1, self.K + 1):
            if k <= self.T:
                self.prob_dist[k - 1] = self.rho / self.T
            else:
                self.prob_dist[k - 1] = (1 - self.rho) / max(1, (self.K - self.T))
        self.prob_dist = self.prob_dist / np.sum(self.prob_dist)

    def choose_bid(self, round_num: int, game=None) -> int:
        return int(np.random.choice(np.arange(1, self.K + 1), p=self.prob_dist))


class EmpiricalFrequencyAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.beta = params.get('beta', 1.0)
        self.empirical_freq = np.ones(K) / K

    def choose_bid(self, round_num: int, game=None) -> int:
        if round_num == 0 or game is None or not hasattr(game, 'agents'):
            return np.random.randint(1, self.K + 1)

        history = game.history if hasattr(game, 'history') else None
        if history is None:
            return np.random.randint(1, self.K + 1)

        N = len(game.agents)
        total_observations = (N - 1) * round_num

        if total_observations > 0 and 'bid_history' in history:
            freq_counts = np.zeros(self.K)

            if hasattr(game, 'agents'):
                for agent in game.agents:
                    if agent.id != self.id and hasattr(agent, 'bid_history'):
                        for bid in agent.bid_history[:round_num]:
                            if 1 <= bid <= self.K:
                                freq_counts[bid - 1] += 1
            elif 'bid_history' in history:
                for agent_id, bids in history['bid_history'].items():
                    if agent_id != self.id:
                        for bid in bids[:round_num]:
                            if 1 <= bid <= self.K:
                                freq_counts[bid - 1] += 1

            if np.sum(freq_counts) > 0:
                self.empirical_freq = freq_counts / total_observations
                self.empirical_freq = np.clip(self.empirical_freq, 1e-6, 1)
                self.empirical_freq = self.empirical_freq / np.sum(self.empirical_freq)

        N_others = N - 1
        P_unique = np.zeros(self.K)
        for k in range(self.K):
            fk = self.empirical_freq[k]
            if N_others >= 1:
                P_unique[k] = N_others * fk * ((1 - fk) ** (N_others - 1))

        k_values = np.arange(1, self.K + 1).astype(float)
        U = P_unique / k_values
        U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)

        if np.sum(U) > 0 and not np.all(U == 0):
            exp_U = np.exp(self.beta * U - np.max(self.beta * U))
            prob_dist = exp_U / np.sum(exp_U)
        else:
            prob_dist = np.ones(self.K) / self.K

        return int(np.random.choice(np.arange(1, self.K + 1), p=prob_dist))

    def reset(self):
        super().reset()
        self.empirical_freq = np.ones(self.K) / self.K


class BestResponseAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.beta = params.get('beta', 1.0)
        self.empirical_dist = np.ones(K) / K

    def choose_bid(self, round_num: int, game=None) -> int:
        if round_num == 0 or game is None:
            return np.random.randint(1, self.K + 1)

        history = game.history if hasattr(game, 'history') else None
        if history is None:
            return np.random.randint(1, self.K + 1)

        N = len(game.agents)
        total_observations = (N - 1) * round_num

        if total_observations > 0:
            dist_counts = np.zeros(self.K)

            if hasattr(game, 'agents'):
                for agent in game.agents:
                    if agent.id != self.id and hasattr(agent, 'bid_history'):
                        for bid in agent.bid_history[:round_num]:
                            if 1 <= bid <= self.K:
                                dist_counts[bid - 1] += 1
            elif 'bid_history' in history:
                for agent_id, bids in history['bid_history'].items():
                    if agent_id != self.id:
                        for bid in bids[:round_num]:
                            if 1 <= bid <= self.K:
                                dist_counts[bid - 1] += 1

            if np.sum(dist_counts) > 0:
                self.empirical_dist = dist_counts / total_observations
                self.empirical_dist = np.clip(self.empirical_dist, 1e-6, 1)
                self.empirical_dist = self.empirical_dist / np.sum(self.empirical_dist)

        N_others = N - 1
        P_win = np.zeros(self.K)

        for k in range(self.K):
            P_no_k = (1 - self.empirical_dist[k]) ** N_others
            P_no_lower_unique = 1.0
            for j in range(k):
                if N_others >= 1:
                    P_j_unique = N_others * self.empirical_dist[j] * ((1 - self.empirical_dist[j]) ** (N_others - 1))
                    P_no_lower_unique *= (1 - P_j_unique)
            P_win[k] = P_no_k * P_no_lower_unique

        P_win = np.nan_to_num(P_win, nan=0.0, posinf=0.0, neginf=0.0)

        if np.sum(P_win) > 0 and not np.all(P_win == 0):
            exp_P = np.exp(self.beta * P_win - np.max(self.beta * P_win))
            prob_dist = exp_P / np.sum(exp_P)
        else:
            prob_dist = np.ones(self.K) / self.K

        return int(np.random.choice(np.arange(1, self.K + 1), p=prob_dist))

    def reset(self):
        super().reset()
        self.empirical_dist = np.ones(self.K) / self.K


class ReinforcementLearningAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.beta = params.get('beta', 1.0)
        self.eta = params.get('eta', 0.1)
        self.Q_values = np.zeros(K)
        self.action_counts = np.zeros(K)

    def choose_bid(self, round_num: int, game=None) -> int:
        if round_num == 0 or np.sum(self.Q_values) == 0:
            return np.random.randint(1, self.K + 1)

        exp_Q = np.exp(self.beta * self.Q_values - np.max(self.beta * self.Q_values))
        prob_dist = exp_Q / np.sum(exp_Q)
        return int(np.random.choice(np.arange(1, self.K + 1), p=prob_dist))

    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        super().update(round_num, all_bids, winner_id, game)
        my_bid = all_bids[self.id]
        reward = 1 if winner_id == self.id else 0
        self.action_counts[my_bid - 1] += 1
        self.Q_values[my_bid - 1] += self.eta * (reward - self.Q_values[my_bid - 1])

    def reset(self):
        super().reset()
        self.Q_values = np.zeros(self.K)
        self.action_counts = np.zeros(self.K)


class ReplicatorDynamicsAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.population_dist = np.ones(K) / K
        self.fitness_history = []

    def choose_bid(self, round_num: int, game=None) -> int:
        return int(np.random.choice(np.arange(1, self.K + 1), p=self.population_dist))

    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        super().update(round_num, all_bids, winner_id, game)
        if round_num > 0:
            fitness = np.zeros(self.K)
            if winner_id is not None:
                winning_bid = all_bids[winner_id]
                fitness[winning_bid - 1] += 1

            avg_fitness = np.sum(self.population_dist * fitness)
            if avg_fitness > 0:
                self.population_dist = self.population_dist * fitness / avg_fitness
                self.population_dist = self.population_dist / np.sum(self.population_dist)

            self.fitness_history.append(fitness.copy())

    def reset(self):
        super().reset()
        self.population_dist = np.ones(self.K) / self.K
        self.fitness_history = []


class LevelKAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.level = params.get('level', 1)
        self.beta = params.get('beta', 1.0)
        self.level_dist = params.get('level_dist', [0.5, 0.3, 0.2])
        self.level_strategies = []
        self._generate_level_strategies()

    def _generate_level_strategies(self):
        self.level_strategies = []
        level0_dist = np.ones(self.K) / self.K
        self.level_strategies.append(level0_dist)

        for L in range(1, self.level + 1):
            mixture_dist = np.zeros(self.K)
            for l in range(L):
                if l < len(self.level_dist):
                    weight = self.level_dist[l] / sum(self.level_dist[:L])
                    mixture_dist += weight * self.level_strategies[l]

            N_others = 5  # as in your original code
            P_win = np.zeros(self.K)
            for k in range(self.K):
                P_no_k = (1 - mixture_dist[k]) ** N_others
                P_no_lower_unique = 1.0
                for j in range(k):
                    P_j_unique = N_others * mixture_dist[j] * ((1 - mixture_dist[j]) ** (N_others - 1))
                    P_no_lower_unique *= (1 - P_j_unique)
                P_win[k] = P_no_k * P_no_lower_unique

            if np.sum(P_win) > 0:
                exp_P = np.exp(self.beta * P_win - np.max(self.beta * P_win))
                levelL_dist = exp_P / np.sum(exp_P)
            else:
                levelL_dist = np.ones(self.K) / self.K

            self.level_strategies.append(levelL_dist)

    def choose_bid(self, round_num: int, game=None) -> int:
        dist = self.level_strategies[self.level] if self.level < len(self.level_strategies) else (np.ones(self.K) / self.K)
        return int(np.random.choice(np.arange(1, self.K + 1), p=dist))


class MutationNoiseAgent(Agent):
    def __init__(self, agent_id: int, K: int, params: Optional[Dict] = None):
        super().__init__(agent_id, K, params)
        self.base_agent_type = params.get('base_agent_type', 'uniform')
        self.mu = params.get('mu', 0.1)

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
        base_bid = self.base_agent.choose_bid(round_num, game)
        if np.random.random() < self.mu:
            return np.random.randint(1, self.K + 1)
        return int(base_bid)

    def update(self, round_num: int, all_bids: Dict[int, int], winner_id: Optional[int], game=None):
        super().update(round_num, all_bids, winner_id, game)
        self.base_agent.update(round_num, all_bids, winner_id, game)

    def reset(self):
        super().reset()
        self.base_agent.reset()


# =========================
# Game
# =========================
class LowestUniqueBidGame:
    def __init__(self, K: int = 10, agents: Optional[List[Agent]] = None, verbose: bool = False, record_dataset: bool = True):
        self.K = K
        self.agents = agents if agents else []
        self.verbose = verbose
        self.record_dataset = record_dataset
        self.round_num = 0

        self.history = {
            'agents': {agent.id: agent for agent in self.agents},
            'bid_history': defaultdict(list),
            'winners': [],
            'all_bids': []
        }

        self.bid_dataset = []
        self.agent_map = {agent.id: agent for agent in self.agents}

        self.bid_counts = np.zeros(K)
        self.win_counts = np.zeros(K)

        self.game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        agent_type_counts = {}
        for agent in self.agents:
            agent_type_counts[type(agent).__name__] = agent_type_counts.get(type(agent).__name__, 0) + 1

        self.metadata = {
            'game_id': self.game_id,
            'K': K,
            'num_agents': len(self.agents),
            'agent_types': agent_type_counts,
            'created_at': datetime.now().isoformat()
        }

    def determine_winner(self, bids: Dict[int, int]) -> Tuple[Optional[int], Optional[int]]:
        freq = Counter(bids.values())
        unique_bids = [bid for bid, count in freq.items() if count == 1]
        if not unique_bids:
            return None, None
        lowest_unique = min(unique_bids)
        winners = [agent_id for agent_id, bid in bids.items() if bid == lowest_unique]
        return (winners[0], lowest_unique) if winners else (None, None)

    def _create_round_record(self, bids: Dict[int, int], winner_id: Optional[int], winning_bid: Optional[int]) -> Dict:
        bid_frequencies = Counter(bids.values())
        record = {
            'game_id': self.game_id,
            'round': self.round_num,
            'timestamp': datetime.now().isoformat(),
            'num_agents': len(self.agents),
            'bids': bids.copy(),
            'bid_frequencies': dict(bid_frequencies),
            'unique_bids': [bid for bid, count in bid_frequencies.items() if count == 1],
            'has_winner': winner_id is not None,
            'winner_id': winner_id,
            'winner_bid': winning_bid,
            'winner_agent_type': None,
            'agent_bids': []
        }
        for agent_id, bid in bids.items():
            agent = self.agent_map[agent_id]
            record['agent_bids'].append({
                'agent_id': agent_id,
                'agent_type': type(agent).__name__,
                'bid': bid,
                'is_winner': agent_id == winner_id,
                'agent_params': agent.params
            })
            if agent_id == winner_id:
                record['winner_agent_type'] = type(agent).__name__
        return record

    def play_round(self) -> Dict:
        bids = {agent.id: agent.choose_bid(self.round_num, self) for agent in self.agents}
        winner_id, winning_bid = self.determine_winner(bids)

        if winner_id is not None:
            self.agent_map[winner_id].points += 1
            self.win_counts[winning_bid - 1] += 1

        for agent in self.agents:
            agent.update(self.round_num, bids, winner_id, self)

        for bid in bids.values():
            self.bid_counts[bid - 1] += 1

        for agent_id, bid in bids.items():
            self.history['bid_history'][agent_id].append(bid)

        self.history['winners'].append(winner_id)
        self.history['all_bids'].append(bids.copy())

        round_record = self._create_round_record(bids, winner_id, winning_bid)
        if self.record_dataset:
            self.bid_dataset.append(round_record)

        if self.verbose:
            print(f"\nRound {self.round_num + 1}:")
            print(f"  Bids: {bids}")
            if winner_id is not None:
                print(f"  Winner: Agent {winner_id} with bid {winning_bid}")
            else:
                print("  No winner (no unique bid)")

        self.round_num += 1
        return round_record

    def play_multiple_rounds(self, num_rounds: int) -> List[Dict]:
        out = []
        for _ in range(num_rounds):
            out.append(self.play_round())
        return out


# =========================
# Agent pool (enforces num_agents!)
# =========================
def create_agent_pool(K: int = 10, num_agents: int = 20) -> List[Agent]:
    agent_configs = [
        ('uniform', 3, {}),
        ('powerlaw', 2, {'alpha': 0.5}),
        ('powerlaw', 2, {'alpha': 1.0}),
        ('powerlaw', 2, {'alpha': 2.0}),
        ('threshold', 2, {'T': max(1, K//3), 'rho': 0.8}),
        ('threshold', 2, {'T': max(1, 2*K//3), 'rho': 0.5}),
        ('empirical', 2, {'beta': 0.5}),
        ('empirical', 1, {'beta': 2.0}),
        ('bestresponse', 2, {'beta': 1.0}),
        ('rl', 2, {'beta': 1.0, 'eta': 0.1}),
        ('replicator', 2, {}),
        ('levelk', 1, {'level': 1, 'beta': 1.0}),
        ('levelk', 1, {'level': 2, 'beta': 1.0}),
        ('mutation', 1, {'base_agent_type': 'empirical', 'mu': 0.1}),
    ]

    agents: List[Agent] = []
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

    # Enforce num_agents (professional + predictable)
    return agents[:num_agents]


# =========================
# Export + plots
# =========================
def build_dataframes(game: LowestUniqueBidGame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rounds = []
    agent_rounds = []

    for rec in game.bid_dataset:
        rounds.append({
            "game_id": rec["game_id"],
            "round": rec["round"],
            "timestamp": rec["timestamp"],
            "num_agents": rec["num_agents"],
            "has_winner": rec["has_winner"],
            "winner_id": rec["winner_id"],
            "winner_bid": rec["winner_bid"],
            "winner_agent_type": rec["winner_agent_type"],
            "unique_count": len(rec["unique_bids"]),
        })

        for a in rec["agent_bids"]:
            agent_rounds.append({
                "game_id": rec["game_id"],
                "round": rec["round"],
                "timestamp": rec["timestamp"],
                "agent_id": a["agent_id"],
                "agent_type": a["agent_type"],
                "bid": a["bid"],
                "is_winner": a["is_winner"],
                "agent_params": json.dumps(a["agent_params"], ensure_ascii=False)
            })

    return pd.DataFrame(rounds), pd.DataFrame(agent_rounds)




def export_outputs_csv(df_rounds: pd.DataFrame, df_agent_rounds: pd.DataFrame, out_dir: str, metadata: dict):
    os.makedirs(out_dir, exist_ok=True)

    # 1) core datasets
    df_rounds.to_csv(os.path.join(out_dir, "rounds.csv"), index=False)
    df_agent_rounds.to_csv(os.path.join(out_dir, "agent_rounds.csv"), index=False)

    # 2) summary as CSV (single-row)
    summary = {
        "total_rounds": int(df_rounds["round"].nunique()) if len(df_rounds) else 0,
        "num_agents": int(df_rounds["num_agents"].iloc[0]) if len(df_rounds) else None,
        "winner_rate": float(df_rounds["has_winner"].mean()) if len(df_rounds) else None,
        "avg_winner_bid": float(df_rounds.loc[df_rounds["has_winner"], "winner_bid"].mean())
            if (len(df_rounds) and df_rounds["has_winner"].any()) else None,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # 3) agent type counts as CSV (multi-row)
    if len(df_agent_rounds):
        type_counts = df_agent_rounds["agent_type"].value_counts().reset_index()
        type_counts.columns = ["agent_type", "count"]
        # make sure plain python ints
        type_counts["count"] = type_counts["count"].astype(int)
    else:
        type_counts = pd.DataFrame(columns=["agent_type", "count"])
    type_counts.to_csv(os.path.join(out_dir, "agent_type_counts.csv"), index=False)

    # 4) metadata as CSV (key/value)
    meta_rows = []
    for k, v in (metadata or {}).items():
        if isinstance(v, dict):
            # flatten dict as JSON-ish string (but still CSV)
            meta_rows.append({"key": k, "value": json.dumps(v, ensure_ascii=False)})
        else:
            meta_rows.append({"key": k, "value": v})
    pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    print("\nSaved outputs to:", out_dir)
    print(" - rounds.csv")
    print(" - agent_rounds.csv")
    print(" - summary.csv")
    print(" - agent_type_counts.csv")
    print(" - metadata.csv")


def make_plots(df_rounds: pd.DataFrame, df_agent_rounds: pd.DataFrame, K: int, out_dir: str):
    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    # 1) Winning bid distribution
    winners = df_rounds[df_rounds["has_winner"]].copy()
    plt.figure()
    if len(winners):
        winners["winner_bid"].value_counts().sort_index().plot(kind="bar")
        plt.xlabel("Winner bid")
        plt.ylabel("Count")
        plt.title("Winning Bid Distribution")
    else:
        plt.text(0.1, 0.5, "No winners in dataset", transform=plt.gca().transAxes)
        plt.title("Winning Bid Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "winning_bid_distribution.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 2) Winner rate over time (rolling)
    plt.figure()
    s = df_rounds["has_winner"].astype(int)
    window = max(5, len(s) // 20) if len(s) else 5
    rolling = s.rolling(window=window, min_periods=1).mean()
    plt.plot(df_rounds["round"], rolling)
    plt.ylim(0, 1)
    plt.xlabel("Round")
    plt.ylabel("Winner rate (rolling)")
    plt.title("Winner Rate Over Time (Rolling Average)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "winner_rate_over_time.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 3) Win rate by agent type
    plt.figure()
    win_by_type = df_agent_rounds.groupby("agent_type")["is_winner"].mean().sort_values(ascending=False)
    win_by_type.plot(kind="bar")
    plt.ylabel("Win rate")
    plt.title("Win Rate by Agent Type")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "win_rate_by_agent_type.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 4) Avg bid by agent type
    plt.figure()
    bid_by_type = df_agent_rounds.groupby("agent_type")["bid"].mean().sort_values()
    bid_by_type.plot(kind="bar")
    plt.ylabel("Average bid")
    plt.title("Average Bid by Agent Type")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "avg_bid_by_agent_type.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 5) Bid distribution by agent type (lines)
    plt.figure()
    for agent_type, sub in df_agent_rounds.groupby("agent_type"):
        counts = sub["bid"].value_counts().reindex(range(1, K + 1), fill_value=0)
        probs = counts / counts.sum() if counts.sum() else counts
        plt.plot(range(1, K + 1), probs.values, label=agent_type)
    plt.xlabel("Bid")
    plt.ylabel("Probability")
    plt.title("Bid Distribution by Agent Type")
    plt.tight_layout()
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(figs_dir, "bid_distribution_by_type.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 6) Cumulative wins for top agents
    wins_by_agent = df_agent_rounds.groupby("agent_id")["is_winner"].sum().sort_values(ascending=False)
    top_agents = list(wins_by_agent.head(5).index)

    plt.figure()
    for aid in top_agents:
        sub = df_agent_rounds[df_agent_rounds["agent_id"] == aid].sort_values("round")
        cumwins = sub["is_winner"].astype(int).cumsum()
        plt.plot(sub["round"], cumwins, label=f"Agent {aid} ({sub['agent_type'].iloc[0]})")
    plt.xlabel("Round")
    plt.ylabel("Cumulative wins")
    plt.title("Top Agents: Cumulative Wins")
    plt.tight_layout()
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(figs_dir, "top_agents_cumwins.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved figures to:", figs_dir)


# =========================
# Main
# =========================
def main():
    # ---- settings (feel free to change) ----
    np.random.seed(42)  # reproducible
    K = 30
    num_agents = 20
    num_rounds = 200     # 10 is demo; 200+ gives meaningful plots
    verbose = False      # True = print every round (noisy)

    # ---- run simulation ----
    agents = create_agent_pool(K=K, num_agents=num_agents)
    game = LowestUniqueBidGame(K=K, agents=agents, verbose=verbose, record_dataset=True)

    print("Starting simulation...")
    game.play_multiple_rounds(num_rounds)

    print(f"\nTotal rounds played: {game.round_num}")
    print(f"Winning prices sample: {[rec['winner_bid'] for rec in game.bid_dataset if rec['has_winner']][:20]}")

    # ---- export + visualize ----
    df_rounds, df_agent_rounds = build_dataframes(game)

    out_dir = os.path.join("outputs", game.game_id)
    export_outputs_csv(df_rounds, df_agent_rounds, out_dir=out_dir, metadata=game.metadata)
    make_plots(df_rounds, df_agent_rounds, K=K, out_dir=out_dir)


if __name__ == "__main__":
    main()
