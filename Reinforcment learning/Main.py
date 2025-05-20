import os
import keyboard
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import RL  # Import your modified TCP-based RL environment
import time

# --- Training Setup ---
REPLAY_BUFFER = './train/1buffer'
CHECKPOINT_DIR = './train/' 
LOG_DIR = './logs/'

game = RL.Game()  

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, '1')
            self.model.save(model_path)
        return True

theCallback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

# --- Game Loop and Hotkeys ---


def pick_case():
    while True:
        key = keyboard.read_key()
        if key == "1":
            return "new"
        elif key == "2":
            return "train"
        elif key == "3":
            return "predict"

cases = pick_case()

def run_game_loop():
    global cases

    match cases:
        case "new":
            model = DQN('MlpPolicy', env=game, tensorboard_log=LOG_DIR, verbose=1, buffer_size=800000, learning_starts=10)
            model.learn(total_timesteps=800000, callback=theCallback)
            model.save_replay_buffer(REPLAY_BUFFER)
        case "train":

            print(f"Loading model from {CHECKPOINT_DIR}")
            #model = DQN('MlpPolicy', env=game, tensorboard_log=LOG_DIR, verbose=1, buffer_size=800000, learning_starts=10)
            model = DQN.load("./train/1", env=game)
            model.load_replay_buffer(REPLAY_BUFFER)
            model.set_env(game)
            model.learn(total_timesteps=20000, callback=theCallback)  # Continue training
        case "predict":
            model = DQN.load("./train/1", env=game)
            model.set_env(game)

            done = False
            obs, _, __= game.get_observation()
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _, info = game.step(action.astype(int).item())
                
                

            print("Prediction finished.")


     

run_game_loop()
