import pygame
import numpy as np
import pickle
import sys
import os
import random

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 165

class NeuralNetwork:
    def __init__(self, input_size=3, hidden_size=6, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b_h = np.random.uniform(-1, 1, (hidden_size, 1))
        self.w_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.b_o = np.random.uniform(-1, 1, (output_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, inputs):
        inputs = np.array(inputs).reshape(-1, 1)
        hidden = self.relu(np.dot(self.w_ih, inputs) + self.b_h)
        output = self.sigmoid(np.dot(self.w_ho, hidden) + self.b_o)
        return output.flatten()[0]

class Bird:
    WIDTH = 34
    HEIGHT = 24
    GRAVITY = 0.5
    FLAP_STRENGTH = 8
    
    def __init__(self, x, y, neural_network):
        self.x = x
        self.y = y
        self.velocity = 0
        self.score = 0
        self.nn = neural_network
        self.alive = True
    
    def update(self, pipes):
        self.velocity += self.GRAVITY
        self.y += self.velocity
        
        if self.y >= SCREEN_HEIGHT or self.y <= 0:
            self.alive = False
            return False
        
        for pipe in pipes:
            if self.check_collision(pipe):
                self.alive = False
                return False
        
        return True
    
    def check_collision(self, pipe):
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
        if not pipes:
            next_pipe = None
        else:
            next_pipe = min(pipes, key=lambda p: abs(p['x'] - self.x))
        
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
        
        output = self.nn.forward(inputs)
        
        if output > 0.5:
            self.velocity = -self.FLAP_STRENGTH
    
    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.WIDTH // 2)

class Pipe:
    WIDTH = 60
    MIN_HEIGHT = 100
    GAP_SIZE = 120
    
    @staticmethod
    def create(x):
        gap_y = random.randint(Pipe.MIN_HEIGHT, SCREEN_HEIGHT - Pipe.MIN_HEIGHT - Pipe.GAP_SIZE)
        return {
            'x': x,
            'width': Pipe.WIDTH,
            'top': gap_y,
            'bottom': gap_y + Pipe.GAP_SIZE,
            'passed': False
        }
    
    @staticmethod
    def update(pipe):
        pipe['x'] -= 5
    
    @staticmethod
    def is_offscreen(pipe):
        return pipe['x'] + pipe['width'] < 0
    
    @staticmethod
    def draw(screen, pipe):
        pygame.draw.rect(screen, (34, 139, 34), (pipe['x'], 0, pipe['width'], pipe['top']))
        pygame.draw.rect(screen, (34, 139, 34), (pipe['x'], pipe['bottom'], pipe['width'], SCREEN_HEIGHT - pipe['bottom']))
    
    @staticmethod
    def check_and_score(bird, pipes):
        """
        Tracking score by checking position of bird and pipe
        score +1 when pipe is to the left of bird
        """
        for pipe in pipes:
            if not pipe['passed']:
                if pipe['x'] + pipe['width'] < bird.x and pipe['x'] < bird.x:
                    if bird.alive:
                        pipe['passed'] = True
                        bird.score += 1

def play_game(neural_network):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - AI Playback (FIXED SCORING)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    
    bird = Bird(50, SCREEN_HEIGHT // 2, neural_network)
    pipes = []
    spawn_counter = 0
    
    running = True
    frame_count = 0
    
    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        spawn_counter += 1
        if spawn_counter > 90:
            pipes.append(Pipe.create(SCREEN_WIDTH))
            spawn_counter = 0
        
        pipes = [p for p in pipes if not Pipe.is_offscreen(p)]
        
        for pipe in pipes:
            Pipe.update(pipe)
        
        # scoring using pipe position
        Pipe.check_and_score(bird, pipes)
        
        bird.think(pipes)
        if not bird.update(pipes):
            print(f"Game Over! Final Score: {bird.score}")
            running = False
        
        screen.fill((135, 206, 235))
        
        for pipe in pipes:
            Pipe.draw(screen, pipe)
        
        bird.draw(screen)
        
        score_text = font.render(f"Score: {bird.score}", True, (0, 0, 0))
        pipes_text = font_small.render(f"Pipes Passed: {bird.score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        screen.blit(pipes_text, (10, 50))
        
        pygame.display.flip()
        frame_count += 1
    
    pygame.quit()
    return bird.score

if __name__ == "__main__":
    model_file = 'best_bird_final.pkl'
    
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    
    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found!")
        print("First run: python flappy_bird_ai.py")
        sys.exit(1)
    
    print(f"Loading model: {model_file}")
    with open(model_file, 'rb') as f:
        nn = pickle.load(f)
    
    print("Watch the bird play! Close window when done.")
    print("(Scoring has been FIXED - no more missed pipes!)")
    score = play_game(nn)
    print(f"Final Score: {score}")
