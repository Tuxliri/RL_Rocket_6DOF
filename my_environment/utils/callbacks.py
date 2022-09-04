from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import Figure

### DEPRECATED ###
class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        self.mydir = None
        super(FigureRecorderCallback, self).__init__(verbose)
    
    def _on_training_start(self) -> None:
        if self.mydir is None:
            self.mydir = os.path.join(
                f"Videos/{self.model.__class__.__name__}", 
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            )
            os.makedirs(self.mydir, exist_ok=True)

        return super()._on_rollout_start()

    def _on_step(self) -> bool:
        self._record_agent_gif(self.model)
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        return super()._on_rollout_end()

    def _record_agent_gif(self, model):
        import imageio
        images = []
        show_plots = False

        env = model.get_env()    

        # Check if environment is vectorized
        env = env.envs[0]

        obs = env.reset()
        img = env.render(mode='rgb_array')

        done = False
        reward_list = []
        rewards_log_list = []
        
        while not done:
            images.append(img)
            action, _ = model.predict(obs)
            obs, rew, done, info = env.step(action)
            reward_list.append(rew)

            rewards_log_list.append(info["rewards_log"])
            img = env.render(mode='rgb_array')

        gif_name = os.path.join(self.mydir, f"lander_{self.n_calls}.gif")
        imageio.mimsave(gif_name, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
        
        fig_rew, ax_rew = plt.subplots()
        fig_rew_time, ax_rew_time = plt.subplots()

        df = pd.DataFrame(rewards_log_list)
        try:
            ax_rew_time.plot(df['time_reward'])
            self.logger.record("Time ep. reward", Figure(fig_rew_time, close=True),
                        exclude=("stdout", "log", "json", "csv"))
            df.drop(['time_reward'], axis=1).plot(
            ax=ax_rew, legend=True
            )
        except:
            pass

        

        states_fig, action_fig = env.plotStates(show_plots)
        env.reset()
        
        # [0] needed as the method returns a list containing the tuple of figures
        # states_fig, action_fig = env.env_method('plotStates', show_fig)[0]
        fig1, ax1 = plt.subplots()
        ax1.set_title('Rewards')
        ax1.plot(reward_list)        

        # Close the figure after logging it
        self.logger.record("Evaluation ep. rewards", Figure(fig1, close=True),
                        exclude=("stdout", "log", "json", "csv"))
        # self.logger.record("Individual ep. rewards", Figure(fig_rew, close=True),
        #                 exclude=("stdout", "log", "json", "csv"))
        
        self.logger.record("States", Figure(states_fig, close=True),
                        exclude=("stdout", "log", "json", "csv"))
        self.logger.record("Thrust", Figure(action_fig, close=True),
                        exclude=("stdout", "log", "json", "csv"))