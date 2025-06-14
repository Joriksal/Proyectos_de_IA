import gym
from gym import spaces
import numpy as np
import random
import cv2
from collections import deque

SNAKE_LEN_GOAL = 30
GRID_SIZE = 500
BLOCK_SIZE = 10
MAX_POS = GRID_SIZE // BLOCK_SIZE

def collision_with_apple():
    return [random.randint(0, MAX_POS - 1) * BLOCK_SIZE,
            random.randint(0, MAX_POS - 1) * BLOCK_SIZE]

def collision_with_boundaries(head):
    return not (0 <= head[0] < GRID_SIZE and 0 <= head[1] < GRID_SIZE)

def collision_with_self(snake):
    return snake[0] in snake[1:]

class SnekEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=False):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)  # 0: Left, 1: Right, 2: Up, 3: Down
        # Observación: imagen RGB del GRID_SIZE x GRID_SIZE
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8
        )

    def reset(self):
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.snake_head = list(self.snake_position[0])
        self.apple_position = collision_with_apple()
        self.score = 0
        self.done = False

        self.prev_actions = deque([-1] * SNAKE_LEN_GOAL, maxlen=SNAKE_LEN_GOAL)

        self.img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)

        self.prev_dist = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self._draw()  # Dibuja la primera imagen

        return self.img.copy()

    def step(self, action):
        last_action = self.prev_actions[-1]
        # Prevenir movimientos inversos
        if (action == 0 and last_action == 1) or (action == 1 and last_action == 0) or \
           (action == 2 and last_action == 3) or (action == 3 and last_action == 2):
            action = last_action

        self.prev_actions.append(action)

        if action == 0:   self.snake_head[0] -= BLOCK_SIZE  # Left
        elif action == 1: self.snake_head[0] += BLOCK_SIZE  # Right
        elif action == 2: self.snake_head[1] -= BLOCK_SIZE  # Up
        elif action == 3: self.snake_head[1] += BLOCK_SIZE  # Down

        apple_reward = 0
        if self.snake_head == self.apple_position:
            self.apple_position = collision_with_apple()
            self.snake_position.insert(0, list(self.snake_head))
            self.score += 1
            apple_reward = 10
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        if collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position):
            self.done = True
            reward = -20
        else:
            dist = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
            dist_reward = (self.prev_dist - dist) * 0.1
            reward = dist_reward + apple_reward
            self.prev_dist = dist

        self._draw()

        return self.img.copy(), reward, self.done, {}

    def _draw(self):
        self.img.fill(0)
        for pos in self.snake_position:
            cv2.rectangle(self.img, tuple(pos), (pos[0] + BLOCK_SIZE, pos[1] + BLOCK_SIZE), (0, 255, 0), -1)
        cv2.rectangle(self.img, tuple(self.apple_position),
                      (self.apple_position[0] + BLOCK_SIZE, self.apple_position[1] + BLOCK_SIZE), (0, 0, 255), -1)

    def render(self, mode="human"):
        if not self.render_mode:
            return
        cv2.imshow("Snake", self.img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            self.done = True
            self.close()

    def close(self):
        if self.render_mode:
            cv2.destroyAllWindows()
