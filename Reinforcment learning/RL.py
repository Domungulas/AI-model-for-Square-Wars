import json
import keyboard.mouse
import numpy as np
import gymnasium as gym
import keyboard
import time
import requests

class Game(gym.Env):
    def __init__(self):
        super().__init__()

        # Maximum number of enemies to track
        self.max_enemies = 5  # Max enemies to track
        self.observation_space = gym.spaces.Box(
            low=-1000, high=1000, shape=(self.max_enemies * 2,), dtype=np.float32
        )  #
        self.action_space = gym.spaces.Discrete(4)

        # Start TCP server
        url = "http://localhost:8000/"
        response = requests.get(url)
        print("Response from Unity:", response.text)

        self.recorded_hp = 100

    def get_observation(self):
        try:
            response = requests.get("http://localhost:8000/")
            game_state = response.json()

            # Extract player position
            player_x = game_state.get("playerX")
            player_y = game_state.get("playerY")

            player_health = game_state.get("playerHP")

            game_over = game_state.get("isGameOver")

            # Extract enemy positions (limit to `max_enemies`)
            enemy_positions = game_state.get("enemyPositions", [])
            enemy_array = []
            for enemy in enemy_positions[:self.max_enemies]:  # Limit to max_enemies
                enemy_array.extend([enemy.get("x", 0), enemy.get("y", 0)])

            # Pad with zeros if fewer enemies
            while len(enemy_array) < self.max_enemies * 2:
                enemy_array.append(990)

            observation = np.array( enemy_array, dtype=np.float32)

            print(observation)

            return observation, game_over, player_health
        except Exception as e:
            print(f"Error receiving data: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    

    def step(self, action):
        reward = 5  # Example reward logic
        terminated = False  # Example termination logic
        truncated = False  # Timeout logic
        info = {}

        action_map = {
            0: 'a',  # Left
            1: 's',  # Down
            2: 'd',  # Right
            3: 'w',  # Up
        }

        # Simulate action by pressing and releasing the appropriate key
        keyboard.press(action_map[action])
        time.sleep(0.15)
        keyboard.release(action_map[action])

        # Check if the game has ended (gameOver flag)
        observation, is_game_over, curr_hp = self.get_observation()

        if curr_hp != self.recorded_hp:
            reward = -20
            self.recorded_hp = curr_hp

        if bool(is_game_over):
            terminated = True
            print("reseted")
            self.reset()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        obs, _, __ = self.get_observation()

        self.recorded_hp = 100

        return obs, {}

    def close(self):
        print("Closing environment")
