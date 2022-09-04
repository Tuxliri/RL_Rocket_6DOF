from imitation.algorithms import bc
import numpy as np
from stable_baselines3.common import policies
import gym
from gym.utils.play import play

class imitationKickstarter():
    def __init__(self, env : gym.Env = None, obs=None, actions=None,
    policy : policies.BasePolicy = None) -> None:
        self.env = env
        self.obs = obs
        self.actions = actions
        self.policy = policy
        self.trajectories = []
        pass

    def play(self, keys_action_map = None, fps=10):
        
        assert self.env is not None, 'You need to provide an environment'

        myCallback = RecordTrajectoryCallback()

        play(
            env=self.env,
            fps=fps,
            callback=myCallback.callback,
            keys_to_action=keys_action_map
            )

        self.trajectories = myCallback.returnTrajectories()

        self.env.close()

        return self.trajectories

    def train(self,n_epochs=1) -> policies.BasePolicy:
        # Train the policy using the trajectories stored in the 
        # self.obs and self.actions attributes
        bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            policy=self.policy,
            demonstrations=self.trajectories
            )

        self.policy = bc_trainer.train(n_epochs=n_epochs)
        
        return bc_trainer.policy
 
class RecordTrajectoryCallback():
    def __init__(self) -> None:
        self.trajectories = []  # List of recorded trajectories containig tuples (obs, acts)
        self.trajectoryObs = []
        self.trajectoryAct = []
        self.trajectoryInfos = []
        self.numTrajectories = 0
        
        pass

    def callback(
        self,
        obs_t,
        obs_tp1,
        action,
        rew,
        done: bool,
        info: dict,
    ):
        self.trajectoryObs.append(obs_t)
        self.trajectoryAct.append(action)
        self.trajectoryInfos.append(info)

        if done:
            self.trajectoryObs.append(obs_tp1)
            
            self.trajectories.append(
                (np.array(self.trajectoryObs),
                np.array(self.trajectoryAct),
                self.trajectoryInfos,
                True) # Terminal flag
                )
            self.trajectoryObs, self.trajectoryAct, self.trajectoryInfos = [], [], []

            self.numTrajectories += 1
        pass

    def returnTrajectories(self):
        from imitation.data.types import Trajectory

        imitationTrajectories = []

        for traj in self.trajectories:
            observations, actions, infos, isterminal = traj
            
            assert observations.shape[0] == actions.shape[0]+1,\
                f"There needs to be {actions.shape[0]+1} observations"\
                    " but there are {observations.shape[0]}"
                    
            imitationTrajectories.append(Trajectory(
                observations,
                actions,
                infos,
                terminal=isterminal)
                )

        return imitationTrajectories
