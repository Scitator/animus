import numpy as np
import torch


class Agent:
    """An agent. Used for model inference.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, acmodel, device, argmax=False, num_envs=1):
        self.acmodel = acmodel
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(
                self.num_envs, self.acmodel.memory_size, device=self.device
            )
        else:
            self.memories = None

        self.acmodel.to(self.device)
        self.acmodel.eval()

    def get_actions(self, obss):
        with torch.no_grad():
            obss = torch.tensor(np.array(obss), device=self.device, dtype=torch.float)
            dist, _, self.memories = self.acmodel(obss, self.memories)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(
                dones, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
