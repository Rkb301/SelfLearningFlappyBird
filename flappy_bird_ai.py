import pygame
import numpy as np
import random
import pickle
import os
import sys
import time

pygame.init()

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 165
SPEED_MULTIPLIER = 1  # Configurable via CLI args

class NeuralNetwork:
    """
    Feedforward Neural Network.
    Architecture: 3 Input -> 6 Hidden -> 1 Output.
    Acts as the 'Brain' for decision making.
    """
    def __init__(self, input_size=3, hidden_size=6, output_size=1):
        # Network dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Init random weights/biases (-1 to 1)
        # w_ih: Input -> Hidden weights
        # b_h: Hidden layer biases
        # w_ho: Hidden -> Output weights
        # b_o: Output layer biases
        self.w_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b_h = np.random.uniform(-1, 1, (hidden_size, 1))
        self.w_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.b_o = np.random.uniform(-1, 1, (output_size, 1))

    def sigmoid(self, x):
        # Activation: Squash to 0-1 range (Probability)
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        # Activation: Zero out negatives (Linearity fix)
        return np.maximum(0, x)

    def forward(self, inputs):
        """
        Forward pass.
        Data flow: Inputs -> Hidden(ReLU) -> Output(Sigmoid).
        Returns flap probability (0.0 - 1.0).
        """
        inputs = np.array(inputs).reshape(-1, 1)
        
        # Hidden layer calc
        hidden = self.relu(np.dot(self.w_ih, inputs) + self.b_h)
        
        # Output layer calc
        output = self.sigmoid(np.dot(self.w_ho, hidden) + self.b_o)
        
        return output.flatten()[0]

    def mutate(self, mutation_rate=0.1):
        """
        Evolution step: Randomly adjust weights.
        Creates variation for next gen.
        """
        # Add Gaussian noise to weights if chance met
        if random.random() < mutation_rate:
            self.w_ih += np.random.normal(0, 0.3, self.w_ih.shape)
        if random.random() < mutation_rate:
            self.b_h += np.random.normal(0, 0.3, self.b_h.shape)
        if random.random() < mutation_rate:
            self.w_ho += np.random.normal(0, 0.3, self.w_ho.shape)
        if random.random() < mutation_rate:
            self.b_o += np.random.normal(0, 0.3, self.b_o.shape)

    def copy(self):
        # Deep copy helper. Avoids reference issues during cloning.
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.w_ih = self.w_ih.copy()
        new_nn.b_h = self.b_h.copy()
        new_nn.w_ho = self.w_ho.copy()
        new_nn.b_o = self.b_o.copy()
        return new_nn

class Bird:
    """
    Bird Agent.
    Contains position, physics, and NN brain.
    """
    WIDTH = 34
    HEIGHT = 24
    GRAVITY = 0.5
    FLAP_STRENGTH = 8
    
    def __init__(self, x, y, neural_network=None):
        # Set initial position, physics state, and brain (NN)
        self.x = x
        self.y = y
        self.velocity = 0
        self.alive = True
        self.score = 0
        self.nn = neural_network if neural_network else NeuralNetwork()
        self.frame_counter = 0  # For debugging
    
    def update(self, pipes):
        # Apply gravity and check if bird died (hit floor/ceiling)
        self.velocity += self.GRAVITY
        self.y += self.velocity
        self.frame_counter += 1
        
        if self.y >= SCREEN_HEIGHT or self.y <= 0:
            self.alive = False
            return
        
        # Check collisions with any active pipes
        for pipe in pipes:
            if self.check_collision(pipe):
                self.alive = False
                return
    
        # Pipe collision check
        for pipe in pipes:
            if self.check_collision(pipe):
                self.alive = False
                return

    def check_collision(self, pipe):
        """AABB Collision check (Rect vs Rect)"""
        bird_left = self.x
        bird_right = self.x + self.WIDTH
        bird_top = self.y
        bird_bottom = self.y + self.HEIGHT
    
        pipe_left = pipe['x']
        pipe_right = pipe['x'] + pipe['width']
        pipe_top = pipe['top']
        pipe_bottom = pipe['bottom']
    
        # Horizontal overlap
        if bird_right > pipe_left and bird_left < pipe_right:
            # Vertical overlap (Top or Bottom pipe)
            if bird_top < pipe_top or bird_bottom > pipe_bottom:
                return True
    
        return False

    def think(self, pipes):
        """
        AI Decision block.
        1. Find target pipe.
        2. Normalize inputs.
        3. Query NN.
        4. Flap if output > 0.5.
        """
        if not pipes:
            next_pipe = None
        else:
            # Closest pipe ahead
            next_pipe = min(pipes, key=lambda p: abs(p['x'] - self.x))
    
        # Input Normalization (0-1 range approx)
        if next_pipe:
            pipe_dist = next_pipe['x'] - self.x
            gap_center = (next_pipe['top'] + next_pipe['bottom']) / 2.0
        else:
            pipe_dist = SCREEN_WIDTH
            gap_center = SCREEN_HEIGHT / 2.0
    
        inputs = [
            self.y / SCREEN_HEIGHT,                  # Normalized Y pos
            (gap_center - self.y) / SCREEN_HEIGHT,   # Vertical dist to gap
            pipe_dist / SCREEN_WIDTH                 # Horizontal dist to pipe
        ]
    
        # NN Forward pass
        output = self.nn.forward(inputs)
    
        # Threshold check
        if output > 0.5:
            self.velocity = -self.FLAP_STRENGTH

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.WIDTH // 2)

class Pipe:
    """Static methods for Pipe logic."""
    WIDTH = 60
    MIN_HEIGHT = 100
    GAP_SIZE = 120

    @staticmethod
    def create(x):
        # Random gap position
        gap_y = random.randint(Pipe.MIN_HEIGHT, SCREEN_HEIGHT - Pipe.MIN_HEIGHT - Pipe.GAP_SIZE)
        return {
            'x': x,
            'width': Pipe.WIDTH,
            'top': gap_y,
            'bottom': gap_y + Pipe.GAP_SIZE,
            'passed': False,        # Score tracking flag
            'frame_created': None
        }

    @staticmethod
    def update(pipe):
        # Move left
        pipe['x'] -= 5
    
    @staticmethod
    def is_offscreen(pipe):
        # Check if pipe has moved past the left edge of screen
        return pipe['x'] + pipe['width'] < 0
    
    @staticmethod
    def draw(screen, pipe):
        # Draw top and bottom parts of the pipe
        pygame.draw.rect(screen, (34, 139, 34), (pipe['x'], 0, pipe['width'], pipe['top']))
        pygame.draw.rect(screen, (34, 139, 34), (pipe['x'], pipe['bottom'], pipe['width'], SCREEN_HEIGHT - pipe['bottom']))
    
    @staticmethod
    def check_and_score(bird, pipes):
        """Updates the score if the bird successfully passes a pipe."""
        for pipe in pipes:
            if not pipe['passed']:
                # Bird has passed through pipe if:
                # 1. Pipe's right edge is to the LEFT of bird's center
                # &
                # 2. Pipe was to the RIGHT of bird before (check frame_created for this)
                if pipe['x'] + pipe['width'] < bird.x and pipe['x'] < bird.x:
                    # Double-check bird did not collide (still alive)
                    if bird.alive:
                        pipe['passed'] = True
                        bird.score += 1
                        print(f"✓ Score +1! Total: {bird.score}")

class Game:
    """
    Main Game Loop & Simulation.
    Handles Speed Multiplier for fast training.
    """
    def __init__(self, speed_multiplier=1):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Flappy Bird - Neural Network Evolution (Speed: {speed_multiplier}x)")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.speed_multiplier = speed_multiplier
        self.base_fps = FPS * speed_multiplier 

    def run_generation(self, birds, generation_num, best_overall_score):
        """
        Run single generation.
        Loop until all birds dead.
        """
        pipes = []
        spawn_counter = 0
        frame_count = 0
        best_gen_score = 0
    
        running = True
        while running:
            # Visual FPS control
            target_fps = self.base_fps if self.speed_multiplier > 1 else FPS
            self.clock.tick(target_fps)
        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, best_overall_score
        
            # Logic Loop: Run multiple times per frame for speedup
            for _ in range(max(1, self.speed_multiplier)):
                spawn_counter += 1
                # Pipe Spawning
                if spawn_counter > 90:
                    new_pipe = Pipe.create(SCREEN_WIDTH)
                    new_pipe['frame_created'] = frame_count
                    pipes.append(new_pipe)
                    spawn_counter = 0
            
                # Pipe Cleanup
                pipes = [pipe for pipe in pipes if not Pipe.is_offscreen(pipe)]
            
                # Pipe Movement
                for pipe in pipes:
                    Pipe.update(pipe)
            
                # Physics & Scoring
                for bird in birds:
                    if bird.alive:
                        Pipe.check_and_score(bird, pipes)
            
                # Check alive count
                alive_birds = [bird for bird in birds if bird.alive]
            
                if not alive_birds:
                    running = False
                    break
            
                # AI Update
                for bird in alive_birds:
                    bird.think(pipes)
                    bird.update(pipes)
            
                frame_count += 1
                # Failsafe: Prevent infinite games
                if frame_count > 5000:
                    running = False
                    break
        
            if not running:
                break
        
            # Render Step (Once per frame)
            self.screen.fill((135, 206, 235))
        
            for pipe in pipes:
                Pipe.draw(self.screen, pipe)
        
            alive_birds = [bird for bird in birds if bird.alive]
            for bird in alive_birds:
                bird.draw(self.screen)
        
            # UI/Stats
            max_score = max((bird.score for bird in birds), default=0)
            best_gen_score = max(best_gen_score, max_score)
            best_overall_score = max(best_overall_score, best_gen_score)
        
            text_gen = self.font_small.render(f"Gen: {generation_num}", True, (0, 0, 0))
            text_alive = self.font_small.render(f"Alive: {len(alive_birds)}/{len(birds)}", True, (0, 0, 0))
            text_best = self.font_small.render(f"Best: {best_overall_score}", True, (0, 0, 0))
            text_curr = self.font_small.render(f"Current: {max_score}", True, (0, 0, 0))
            text_speed = self.font_small.render(f"Speed: {self.speed_multiplier}x", True, (255, 100, 0))
        
            self.screen.blit(text_gen, (10, 10))
            self.screen.blit(text_alive, (10, 35))
            self.screen.blit(text_best, (10, 60))
            self.screen.blit(text_curr, (10, 85))
            self.screen.blit(text_speed, (10, 110))
        
            pygame.display.flip()
    
        return alive_birds, best_overall_score

class Population:
    """
    Genetic Algorithm Manager.
    Handles population lifecycle: Create -> Evaluate -> Evolve.
    """
    def __init__(self, size=50, speed_multiplier=1):
        self.size = size
        self.birds = [Bird(50, SCREEN_HEIGHT // 2) for _ in range(size)]
        self.generation = 0
        self.best_score = 0
        self.game = Game(speed_multiplier=speed_multiplier)

    def evaluate(self):
        """Run simulation for current gen."""
        survivors, best_score = self.game.run_generation(self.birds, self.generation, self.best_score)
        self.best_score = best_score
    
        if survivors is None:
            return False
    
        return True

    def evolve(self):
        """
        Evolution Step.
        1. Select best.
        2. Clone Elites.
        3. Mutate remainder.
        """
        # Sort by fitness
        self.birds.sort(key=lambda b: b.score, reverse=True)
    
        # Elitism: Top 25%
        elite = self.birds[:max(2, self.size // 4)]
    
        new_generation = []
        # Clone elites
        for bird in elite:
            new_generation.append(Bird(50, SCREEN_HEIGHT // 2, bird.nn.copy()))
    
        # Fill rest via mutation
        while len(new_generation) < self.size:
            parent = random.choice(elite)
            child_nn = parent.nn.copy()
            child_nn.mutate(mutation_rate=0.3)
            new_generation.append(Bird(50, SCREEN_HEIGHT // 2, child_nn))
    
        self.birds = new_generation
        self.generation += 1

def main():
    # Config defaults
    speed_multiplier = 1
    max_gens = 5000
    
    # CLI Args parsing
    for arg in sys.argv[1:]:
        if arg.startswith('--speed='):
            speed_multiplier = int(arg.split('=')[1])
        elif arg.startswith('-s'):
            speed_multiplier = int(arg[2:])
        elif arg == '--fast':
            max_gens = 50
        elif arg == '--help':
            print("""
Flappy Bird AI Training

Usage:
    python flappy_bird_ai.py [options]

Options:
    --speed=N, -sN    Speed multiplier (1-10)
    --fast            Run only 50 generations
    --help            Show this help
            """)
            return

    print(f"Starting Flappy Bird AI Evolution")
    print(f"Speed Multiplier: {speed_multiplier}x")
    print(f"Max Generations: {max_gens}")
    print("=" * 60)

    # Create init population
    population = Population(size=30, speed_multiplier=speed_multiplier)

    # Main Loop
    for gen_num in range(max_gens):
        print(f"Generation {population.generation}: Best Score = {population.best_score}")
    
        # 1. Play
        if not population.evaluate():
            print("Training interrupted by user")
            break
    
        # 2. Evolve
        population.evolve()
    
        # Checkpoint every 10 gens
        if population.generation % 10 == 0:
            best_bird = max(population.birds, key=lambda b: b.score)
            filename = f'best_bird_gen_{population.generation}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(best_bird.nn, f)
            print(f"  Saved checkpoint: {filename}")

    print(f"\n{'='*60}")
    print(f"Training Complete! Best Score: {population.best_score}")
    print(f"{'='*60}")

    # Save final
    best_bird = max(population.birds, key=lambda b: b.score)
    with open('best_bird_final.pkl', 'wb') as f:
        pickle.dump(best_bird.nn, f)
    print("Saved final model: best_bird_final.pkl")

    pygame.quit()

if __name__ == "__main__":
    main()
