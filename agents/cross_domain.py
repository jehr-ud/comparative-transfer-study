from ray.rllib.algorithms.algorithm import Algorithm


class CrossDomainTransfer:
    def __init__(self, policy: Algorithm):
        self.policy = policy

    def predict(self, observation, **kwargs):
        # Predict actions for multi-agent observation
        actions = {}
        for agent_id, obs in observation.items():
            actions[agent_id] = self.policy.compute_single_action(obs, **kwargs)[0]
        return actions

    @classmethod
    def load(cls, model_path: str):
        """Load a previously
            saved MARWIL agent and wrap it in CrossDomainTransfer."""
        policy = Algorithm.from_checkpoint(model_path)
        return cls(policy)
