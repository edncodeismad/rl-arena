import random
from enum import Enum
from collections import deque, namedtuple

class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.num_blocks_w = self.w // BLOCK_SIZE
        self.num_blocks_h = self.h // BLOCK_SIZE
        self.reset()

    def reset(self):
        self.frame_count = 0
        self.direction = Direction.RIGHT.value
        
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = deque([
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ])
        self.snake_set = set(self.snake)
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        all_points = set(Point(x, y) for x in range(0, self.w, BLOCK_SIZE) for y in range(0, self.h, BLOCK_SIZE))
        available_points = list(all_points - self.snake_set)
        self.food = random.choice(available_points)
        
    def play_step(self, action):
        self.frame_count += 1
        
        # Move the snake
        self._move(action)
        self.snake.appendleft(self.head)
        self.snake_set.add(self.head)
        
        # Check for collision
        if self._is_collision() or self.frame_count > 100 * len(self.snake):
            return -10.0, True, self.score  # Negative reward, done, score
        
        # Check if food is eaten
        reward = 0.0
        if self.head == self.food:
            reward = 10.0
            self.score += 1
            self._place_food()
        else:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
        
        return reward, False, self.score
    
    def _is_collision(self, point=None):
        if point is None:
            point = self.head
        return (
            point.x < 0 or point.x >= self.w or
            point.y < 0 or point.y >= self.h or
            point in self.snake_set
        )
        
    def _move(self, action):
        x, y = self.head.x, self.head.y
        
        # Change direction based on action
        if action == 1:  # Turn right
            self.direction = (self.direction % 4) + 1
        elif action == 2:  # Turn left
            self.direction = (self.direction - 2) % 4 + 1
        
        # Update head position
        if self.direction == Direction.RIGHT.value:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT.value:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN.value:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP.value:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)
