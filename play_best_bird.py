import pygame
import numpy as np
import pickle
import sys
import os
import random

pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 165

class NeuralNetwork:
    """
    Neural Network (Recreated for Playback).
    Loads weights from file. Defines decision structure.
    Must match training architecture exactly.
    """
    def __init__(self, input_size=3, hidden_size=6, output_size=1):
        # Structure init (3 In -> 6 Hidden -> 1 Out)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights/Biases matrices
        # Values will be overwritten by pickle.load()
        self.w_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b_h = np.random.uniform(-1, 1, (hidden_size, 1))
        self.w_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.b_o = np.random.uniform(-1, 1, (output_size, 1))

    def sigmoid(self, x):
        # Activation: Squash to 0-1 (Probability)
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        # Activation: Zero out negatives (Linearity)
        return np.maximum(0, x)

    def forward(self, inputs):
        """
        Forward Pass.
        Data path: Input -> Hidden(ReLU) -> Output(Sigmoid).
        Returns decision probability.
        """
        inputs = np.array(inputs).reshape(-1, 1)
        
        # Hidden layer math
        hidden = self.relu(np.dot(self.w_ih, inputs) + self.b_h)
        
        # Output layer math
        output = self.sigmoid(np.dot(self.w_ho, hidden) + self.b_o)
        
        return output.flatten()[0]

class Bird:
    """
    Bird Agent (Playback Mode).
    Single instance controlled by loaded brain.
    """
    WIDTH = 34
    HEIGHT = 24
    GRAVITY = 0.5
    FLAP_STRENGTH = 8

    def __init__(self, x, y, neural_network):
        # Position/Physics init
        self.x = x
        self.y = y
        self.velocity = 0
        self.score = 0
        
        # Assign loaded brain
        self.nn = neural_network
        self.alive = True

    def update(self, pipes):
        """Update physics and check death conditions."""
        # Gravity application
        self.velocity += self.GRAVITY
        self.y += self.velocity
    
        # Boundary check (Floor/Ceiling)
        if self.y >= SCREEN_HEIGHT or self.y <= 0:
            self.alive = False
            return False
    
        # Pipe collision check
        for pipe in pipes:
            if self.check_collision(pipe):
                self.alive = False
                return False
    
        return True

    def check_collision(self, pipe):
        """AABB Collision (Rect overlap check)."""
        bird_left = self.x
        bird_right = self.x + self.WIDTH
        bird_top = self.y
        bird_bottom = self.y + self.HEIGHT
    
        pipe_left = pipe['x']
        pipe_right = pipe['x'] + pipe['width']
        pipe_top = pipe['top']
        pipe_bottom = pipe['bottom']
    
        # Horizontal intersection
        if bird_right > pipe_left and bird_left < pipe_right:
            # Vertical intersection (Hit top OR bottom pipe)
            if bird_top < pipe_top or bird_bottom > pipe_bottom:
                return True
    
        return False

    def think(self, pipes):
        """
        AI Decision Block.
        1. Identify target pipe.
        2. Normalize inputs.
        3. Query Brain.
        4. Flap if confident.
        """
        if not pipes:
            next_pipe = None
        else:
            # Find nearest pipe
            next_pipe = min(pipes, key=lambda p: abs(p['x'] - self.x))
    
        # Normalize Inputs (0-1 range)
        if next_pipe:
            pipe_dist = next_pipe['x'] - self.x
            gap_center = (next_pipe['top'] + next_pipe['bottom']) / 2.0
        else:
            pipe_dist = SCREEN_WIDTH
            gap_center = SCREEN_HEIGHT / 2.0
    
        inputs = [
            self.y / SCREEN_HEIGHT,                  # Normalized Y
            (gap_center - self.y) / SCREEN_HEIGHT,   # Vertical dist to gap
            pipe_dist / SCREEN_WIDTH                 # Horizontal dist to pipe
        ]
    
        # Get network decision
        output = self.nn.forward(inputs)
    
        # Threshold check (> 50% confidence)
        if output > 0.5:
            self.velocity = -self.FLAP_STRENGTH

    def draw(self, screen):
        # Draw bird (Yellow Circle)
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.WIDTH // 2)

class Pipe:
    """Pipe Logic (Generation, movement, rendering)."""
    WIDTH = 60
    MIN_HEIGHT = 100
    GAP_SIZE = 120

    @staticmethod
    def create(x):
        # Random gap y-position
        gap_y = random.randint(Pipe.MIN_HEIGHT, SCREEN_HEIGHT - Pipe.MIN_HEIGHT - Pipe.GAP_SIZE)
        return {
            'x': x,
            'width': Pipe.WIDTH,
            'top': gap_y,
            'bottom': gap_y + Pipe.GAP_SIZE,
            'passed': False  # Prevents double scoring
        }

    @staticmethod
    def update(pipe):
        # Scroll left
        pipe['x'] -= 5

    @staticmethod
    def is_offscreen(pipe):
        # Cleanup check
        return pipe['x'] + pipe['width'] < 0

    @staticmethod
    def draw(screen, pipe):
        # Draw top and bottom rects (Green)
        pygame.draw.rect(screen, (34, 139, 34), (pipe['x'], 0, pipe['width'], pipe['top']))
        pygame.draw.rect(screen, (34, 139, 34), (pipe['x'], pipe['bottom'], pipe['width'], SCREEN_HEIGHT - pipe['bottom']))

    @staticmethod
    def check_and_score(bird, pipes):
        """Score update logic."""
        for pipe in pipes:
            if not pipe['passed']:
                # Check if pipe fully passed bird
                if pipe['x'] + pipe['width'] < bird.x and pipe['x'] < bird.x:
                    if bird.alive:
                        pipe['passed'] = True
                        bird.score += 1

def play_game(neural_network):
    """
    Main Game Loop (Playback Mode).
    Single bird instance. No speedup.
    """
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - AI Playback (FIXED SCORING)")
    clock = pygame.time.Clock()
    
    # Font setup
    font = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)

    # Init bird with loaded brain
    bird = Bird(50, SCREEN_HEIGHT // 2, neural_network)
    pipes = []
    spawn_counter = 0

    running = True
    frame_count = 0

    # --- Game Loop ---
    while running:
        clock.tick(FPS)
    
        # Event Check
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
        # Pipe Spawning (Every 90 frames)
        spawn_counter += 1
        if spawn_counter > 90:
            pipes.append(Pipe.create(SCREEN_WIDTH))
            spawn_counter = 0
    
        # Pipe Cleanup
        pipes = [p for p in pipes if not Pipe.is_offscreen(p)]
    
        # Pipe Update
        for pipe in pipes:
            Pipe.update(pipe)
    
        # Score Update
        Pipe.check_and_score(bird, pipes)
    
        # AI Update
        bird.think(pipes)
        if not bird.update(pipes):
            print(f"Game Over! Final Score: {bird.score}")
            running = False
    
        # Render
        screen.fill((135, 206, 235))
    
        for pipe in pipes:
            Pipe.draw(screen, pipe)
    
        bird.draw(screen)
    
        # Stats Display
        score_text = font.render(f"Score: {bird.score}", True, (0, 0, 0))
        pipes_text = font_small.render(f"Pipes Passed: {bird.score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        screen.blit(pipes_text, (10, 50))
    
        pygame.display.flip()
        frame_count += 1

    pygame.quit()
    return bird.score

if __name__ == "__main__":
    # CLI Entry point
    
    # Default model
    model_file = 'best_bird_final.pkl'

    # Optional: Override model via arg
    if len(sys.argv) > 1:
        model_file = sys.argv[1]

    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found!")
        print("First run: python flappy_bird_ai.py")
        sys.exit(1)

    # Load Model
    print(f"Loading model: {model_file}")
    with open(model_file, 'rb') as f:
        nn = pickle.load(f)

    # Start Game
    print("Watch the bird play! Close window when done.")
    print("(Scoring has been FIXED - no more missed pipes!)")
    
    score = play_game(nn)
    
    print(f"Final Score: {score}")
