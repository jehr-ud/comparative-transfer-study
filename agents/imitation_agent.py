from ray.rllib.algorithms.marwil import MARWIL, MARWILConfig


class ImitationTransfer:
    def __init__(self, policy):
        self.policy = policy

    def predict(self, observation):
        action = self.policy.compute_single_action(observation)
        return action, None

    @classmethod
    def load(cls, model_path):
        config = MARWILConfig().framework("torch").rollouts(num_rollout_workers=0)
        algo = MARWIL(config)
        algo.restore(model_path)
        return cls(policy=algo.get_policy(), path=model_path)
