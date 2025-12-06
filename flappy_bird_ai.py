import pygame
import numpy as np
import random
import pickle
import os
import sys
import time

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 165
SPEED_MULTIPLIER = 1  # Will be set via command line args at runtime

class NeuralNetwork:
    """
    Feedforward neural network with
    3 inputs (game state),
    6 hidden (hidden layer),
    1 output (flap or not flap)
    """
    def __init__(self, input_size=3, hidden_size=6, output_size=1):
        # Initialize weights and biases with random values
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b_h = np.random.uniform(-1, 1, (hidden_size, 1))
        self.w_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.b_o = np.random.uniform(-1, 1, (output_size, 1))

    def sigmoid(self, x):
        # Activation function to squash output between 0 and 1
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        # Activation function: returns x if positive, else 0
        return np.maximum(0, x)

    def forward(self, inputs):
        # Calculate the output of the network given inputs
        inputs = np.array(inputs).reshape(-1, 1)
        hidden = self.relu(np.dot(self.w_ih, inputs) + self.b_h)
        output = self.sigmoid(np.dot(self.w_ho, hidden) + self.b_o)
        return output.flatten()[0]

    def mutate(self, mutation_rate=0.1):
        # Randomly adjust weights and biases to create variations
        if random.random() < mutation_rate:
            self.w_ih += np.random.normal(0, 0.3, self.w_ih.shape)
        if random.random() < mutation_rate:
            self.b_h += np.random.normal(0, 0.3, self.b_h.shape)
        if random.random() < mutation_rate:
            self.w_ho += np.random.normal(0, 0.3, self.w_ho.shape)
        if random.random() < mutation_rate:
            self.b_o += np.random.normal(0, 0.3, self.b_o.shape)
    
    def copy(self):
        # Create an exact clone of this neural network
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.w_ih = self.w_ih.copy()
        new_nn.b_h = self.b_h.copy()
        new_nn.w_ho = self.w_ho.copy()
        new_nn.b_o = self.b_o.copy()
        return new_nn

class Bird:
    """
    Bird class,
    it uses the neural network to control its movement
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
    
    def check_collision(self, pipe):
        # Calculate bounding boxes to detect overlaps
        bird_left = self.x
        bird_right = self.x + self.WIDTH
        bird_top = self.y
        bird_bottom = self.y + self.HEIGHT
        
        pipe_left = pipe['x']
        pipe_right = pipe['x'] + pipe['width']
        pipe_top = pipe['top']
        pipe_bottom = pipe['bottom']
        
        if bird_right > pipe_left and bird_left < pipe_right:
            if bird_top < pipe_top or bird_bottom > pipe_bottom:
                return True
        
        return False
    
    def think(self, pipes):
        # Find the closest pipe to make a decision
        if not pipes:
            next_pipe = None
        else:
            next_pipe = min(pipes, key=lambda p: abs(p['x'] - self.x))
        
        # Determine inputs for the neural network
        if next_pipe:
            pipe_dist = next_pipe['x'] - self.x
            gap_center = (next_pipe['top'] + next_pipe['bottom']) / 2.0
        else:
            pipe_dist = SCREEN_WIDTH
            gap_center = SCREEN_HEIGHT / 2.0
        
        inputs = [
            self.y / SCREEN_HEIGHT,
            (gap_center - self.y) / SCREEN_HEIGHT,
            pipe_dist / SCREEN_WIDTH
        ]
        
        # Feed inputs to NN and jump if output > 0.5
        output = self.nn.forward(inputs)
        
        if output > 0.5:
            self.velocity = -self.FLAP_STRENGTH
    
    def draw(self, screen):
        # Render the bird on screen
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.WIDTH // 2)

class Pipe:
    """
    Handles creating pipes with random gaps and checking if they're off-screen.
    """
    WIDTH = 60
    MIN_HEIGHT = 100
    GAP_SIZE = 120
    
    @staticmethod
    def create(x):
        # Generate a new pipe with random gap height
        gap_y = random.randint(Pipe.MIN_HEIGHT, SCREEN_HEIGHT - Pipe.MIN_HEIGHT - Pipe.GAP_SIZE)
        return {
            'x': x,
            'width': Pipe.WIDTH,
            'top': gap_y,
            'bottom': gap_y + Pipe.GAP_SIZE,
            'passed': False,
            'frame_created': None  # Track when pipe created, for scoring
        }
    
    @staticmethod
    def update(pipe):
        # Move pipe to the left
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
    Main game class. Handles the game loop, drawing, and the speed multiplier
    that lets us train the AI much faster
    """
    def __init__(self, speed_multiplier=1):
        # Setup Pygame window, fonts, and timing
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Flappy Bird - Neural Network Evolution (Speed: {speed_multiplier}x)")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.speed_multiplier = speed_multiplier
        self.base_fps = FPS * speed_multiplier
    
    def run_generation(self, birds, generation_num, best_overall_score):
        """Runs a single generation of birds until they all crash/collide with pipes"""
        # Initialize variables for this run
        pipes = []
        spawn_counter = 0
        frame_count = 0
        best_gen_score = 0
        
        running = True
        while running:
            # Limit frame rate for visualization
            # Speed multiplier: skip frames in rendering loop
            # Lower FPS cap = visual speedup
            target_fps = self.base_fps if self.speed_multiplier > 1 else FPS
            self.clock.tick(target_fps)
        
            # Handle window close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, best_overall_score
        
            # Game Logic Loop - runs multiple times per frame if speed > 1
            # Logic update loop - runs multiple times per frame if speed_multiplier > 1
            for _ in range(max(1, self.speed_multiplier)):
                spawn_counter += 1
                # Spawn new pipe every 90 frames
                if spawn_counter > 90:
                    new_pipe = Pipe.create(SCREEN_WIDTH)
                    new_pipe['frame_created'] = frame_count
                    pipes.append(new_pipe)
                    spawn_counter = 0
                
                # Remove off-screen pipes
                pipes = [pipe for pipe in pipes if not Pipe.is_offscreen(pipe)]
                
                # Move pipes
                for pipe in pipes:
                    Pipe.update(pipe)
                
                # Check collisions and update scores
                for bird in birds:
                    if bird.alive:
                        Pipe.check_and_score(bird, pipes)
                
                # Get list of currently alive birds
                alive_birds = [bird for bird in birds if bird.alive]
                
                # If everyone is dead, end the generation
                if not alive_birds:
                    running = False
                    break
                
                # Let birds think and move
                for bird in alive_birds:
                    bird.think(pipes)
                    bird.update(pipes)
                
                frame_count += 1
                # Safety stop to prevent infinite loops if birds get too good
                if frame_count > 5000:
                    running = False
                    break
        
            if not running:
                break
        
            # Render everything (only done once per frame loop, regardless of speed multiplier)
            # Render at normal speed
            self.screen.fill((135, 206, 235))
        
            for pipe in pipes:
                Pipe.draw(self.screen, pipe)
            
            alive_birds = [bird for bird in birds if bird.alive]
            for bird in alive_birds:
                bird.draw(self.screen)
            
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
    Manages the population of birds
    Handles the evolutionary algorithm parts: selection, mutation, and reproduction
    """
    def __init__(self, size=50, speed_multiplier=1):
        # Initialize population of birds and game instance
        self.size = size
        self.birds = [Bird(50, SCREEN_HEIGHT // 2) for _ in range(size)]
        self.generation = 0
        self.best_score = 0
        self.game = Game(speed_multiplier=speed_multiplier)
    
    def evaluate(self):
        """Runs the game for the current generation to see how well they do"""
        survivors, best_score = self.game.run_generation(self.birds, self.generation, self.best_score)
        self.best_score = best_score
        
        if survivors is None:
            return False
        
        return True
    
    def evolve(self):
        """
        Create next generation through selection and mutation (keep best birds and mutate them)
        """
        # Sort birds by score to find the best ones
        self.birds.sort(key=lambda b: b.score, reverse=True)
    
        # Keep top performers (Elitism)
        elite = self.birds[:max(2, self.size // 4)]
    
        new_generation = []
        # Add elites directly to new generation
        for bird in elite:
            new_generation.append(Bird(50, SCREEN_HEIGHT // 2, bird.nn.copy()))
    
        # Fill the rest of population by mutating copies of elites
        while len(new_generation) < self.size:
            parent = random.choice(elite)
            child_nn = parent.nn.copy()
            child_nn.mutate(mutation_rate=0.3)
            new_generation.append(Bird(50, SCREEN_HEIGHT // 2, child_nn))
    
        self.birds = new_generation
        self.generation += 1

def main():
    # Parse command line arguments
    speed_multiplier = 1
    max_gens = 5000
    
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
                      1 = normal speed
                      5 = 5x faster visuals
                      10 = 10x faster (max)
    --fast            50 generations instead of 100
    --help            Show this help

Examples:
    python flappy_bird_ai.py              # Normal speed (decent results but slowest!)
    python flappy_bird_ai.py --speed=5    # 5x faster visuals (best results and fast!)
    python flappy_bird_ai.py -s10         # 10x faster (MAX!) (not reliable!)
    python flappy_bird_ai.py -s5 --fast   # 5x speed + 50 gens (decently reliable!)
            """)
            return
    
    print(f"Starting Flappy Bird AI Evolution")
    print(f"Speed Multiplier: {speed_multiplier}x")
    print(f"Max Generations: {max_gens}")
    print("=" * 60)

    # Create initial population
    population = Population(size=30, speed_multiplier=speed_multiplier)

    # Main evolution loop
    for gen_num in range(max_gens):
        print(f"Generation {population.generation}: Best Score = {population.best_score}")
    
        # Run game for this generation
        if not population.evaluate():
            print("Training interrupted by user")
            break
    
        # Create next generation
        population.evolve()
    
        # Save best bird every 10 generations
        if population.generation % 10 == 0:
            best_bird = max(population.birds, key=lambda b: b.score)
            filename = f'best_bird_gen_{population.generation}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(best_bird.nn, f)
            print(f"  Saved checkpoint: {filename}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Score: {population.best_score}")
    print(f"{'='*60}")

    # Save final best bird
    best_bird = max(population.birds, key=lambda b: b.score)
    with open('best_bird_final.pkl', 'wb') as f:
        pickle.dump(best_bird.nn, f)
    print("Saved final model: best_bird_final.pkl")

    pygame.quit()

if __name__ == "__main__":
    main()
