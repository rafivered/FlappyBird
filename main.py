import pickle
from typing import Dict, Any

import pygame
import random
from simple_neural_network import SimpleNeuralNetwork, nn
random.seed(0)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



class fluppy_bird:
    def __init__(self, file_path = None):
        pygame.init()
        self.game_active = False
        self.score = 0
        self.best_score = 0
        self.generation = 0

        self.gravity = 0.5
        self.number_of_birds = 1000
        self.obstacle_speed = 5
        self.obstacle_color = (0, 128, 0)  # Green
        self.mean_obstacle_distance = 400

        self.screen_width = 1000
        self.screen_height = 600
        self.font = pygame.font.Font(None, 36)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flappy Bird Game")
        self.clock = pygame.time.Clock()

        self.neural_networks = [] #[SimpleNeuralNetwork() for _ in range(self.number_of_birds)]
        if file_path is not None:
            self.neural_networks.append(SimpleNeuralNetwork())
            self.neural_networks[0].load_from_memory()




    def init_game(self, best_nn = None):
        self.birds = [
            {'x': 50, 'y': 300, 'color': self.generate_random_bird_color(), 'bird_radius': self.generate_random_bird_radius(),
             'movement': 0, 'alive': True, 'score': 0} for _ in range(self.number_of_birds)]

        self.obstacles = [{'x': self.screen_width, 'width': 70, 'height': random.randint(150, 450), 'gap': 200}]
        self.generation += 1

        if len(self.neural_networks) == 1:
            best_nn = [0]

        if best_nn is None:
            self.neural_networks = [SimpleNeuralNetwork() for _ in range(self.number_of_birds)]
        else:
            nn_list = [self.neural_networks[i] for i in best_nn]
            self.neural_networks = [SimpleNeuralNetwork(nn_list) for _ in range(self.number_of_birds - len(nn_list))]
            self.neural_networks = nn_list + self.neural_networks
            if self.score > 100000:
                nn_list[0].save_to_memory()

        self.score = 0
        self.game_active = True

    def update_game(self):
        all_birds_dead = True

        for index, bird in enumerate(self.birds):
            if bird['alive']:
                bird['score'] = self.score
                # The neural network decides whether to jump or not
                inputs = self.get_nn_inputs(bird)
                decision = nn(self.neural_networks[index], *inputs)
                if decision > 0.5:  # Threshold for jumping
                    bird['movement'] = -10  # Bird flaps

                # Update bird position
                bird['movement'] += self.gravity
                bird['y'] += bird['movement']

                # Check for collisions
                if bird['y'] > self.screen_height - bird['bird_radius'] or bird['y'] < 0 or \
                        self.obstacles[0]['x'] < bird['x'] + bird['bird_radius'] < self.obstacles[0]['x'] + self.obstacles[0][
                    'width'] and \
                        (bird['y'] < self.obstacles[0]['height'] or bird['y'] > self.obstacles[0]['height'] + self.obstacles[0][
                            'gap']):
                    bird['alive'] = False
                else:
                    all_birds_dead = False

        for index, obstacle in enumerate(self.obstacles):
            obstacle['x'] -= self.obstacle_speed

            if obstacle['x'] < -obstacle['width']:
                self.obstacles.remove(obstacle)
                self.score += 1
                # if score is a multiple of 10, increase the speed of the obstacle
                if self.score % 10 == 0:
                    self.obstacle_speed += 0

        obstacle_distance = self.mean_obstacle_distance  + round(random.uniform(-10, 10))
        if self.screen_width - self.obstacles[-1]['x'] + self.obstacles[-1]['width'] >= obstacle_distance:
            self.obstacles.append(
                {'x': self.screen_width, 'width': 70, 'height': random.randint(0, self.screen_height - 200), 'gap': 200})

        if len([bird for bird in self.birds if bird['alive']]) <= -1:
            self.game_active = False
            self.best_score = max(self.score, self.best_score)
            best_neural_networks_indx = [indx for indx, bird in enumerate(self.birds) if bird['alive']]
            return best_neural_networks_indx

        if all_birds_dead:
            self.best_score = max(self.score, self.best_score)
            sorted_list = sorted(enumerate(self.birds), key=lambda x: x[1]['score'], reverse=True)
            best_neural_networks_indx = [index for index, _ in sorted_list[:5]]
            self.game_active = False
            return best_neural_networks_indx

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if pygame.KEYDOWN == event.type and event.key == pygame.K_r and not self.game_active:
                self.init_game()

    def get_nn_inputs(self, bird):
        x_distance_from_1st_obstacle = max(self.obstacles[0]['x'] - bird['x'], 0)
        y_distance_from_1st_lower_obstacle = self.obstacles[0]['height'] + self.obstacles[0]['gap'] - bird['y']
        x_distance_from_2nd_obstacle = 0
        y_distance_from_2nd_lower_obstacle = 0
        x_distance_from_3rd_obstacle = 0
        y_distance_from_3rd_lower_obstacle = 0
        if len(self.obstacles) > 1:
            x_distance_from_2nd_obstacle = max(self.obstacles[1]['x'] - bird['x'], 0)
            y_distance_from_2nd_lower_obstacle = self.obstacles[1]['height'] + self.obstacles[1]['gap'] - bird['y']
            if len(self.obstacles) > 2:
                x_distance_from_3rd_obstacle = max(self.obstacles[2]['x'] - bird['x'], 0)
                y_distance_from_3rd_lower_obstacle = self.obstacles[2]['height'] + self.obstacles[2]['gap'] - bird['y']

        current_movement = bird['movement']
        y_distance_from_top = bird['y']
        y_distance_from_bottom = self.screen_height - bird['y']
        bird_radius = bird['bird_radius']
        return [x_distance_from_1st_obstacle, y_distance_from_1st_lower_obstacle, x_distance_from_2nd_obstacle,
                y_distance_from_2nd_lower_obstacle, x_distance_from_3rd_obstacle, y_distance_from_3rd_lower_obstacle,
                current_movement, y_distance_from_top,
                y_distance_from_bottom, self.obstacle_speed, bird_radius]

    def draw_game(self):
        self.screen.fill((135, 206, 250))  # Light blue background
        bird: dict[str, int | bool | Any]
        for bird in self.birds:
            if bird['alive']:
                self.draw_bird(bird['x'], bird['y'], bird['color'], bird['bird_radius'])
        for obstacle in self.obstacles:
            self.draw_obstacle(obstacle['x'], obstacle['width'], obstacle['height'], obstacle['gap'])

        if self.game_active:
            self.draw_score()
            self.draw_generation()
            self.draw_leading_birds_id()

        else:
            self.draw_game_over()

    def draw_bird(self, x, y, bird_color, bird_radius):
        pygame.draw.circle(self.screen, bird_color, (x, int(y)), bird_radius)

    def draw_obstacle(self, x, width, height, gap):
        pygame.draw.rect(self.screen, self.obstacle_color, (x, 0, width, height))
        pygame.draw.rect(self.screen, self.obstacle_color,
                         (x, height + gap, width, self.screen_height))

    def draw_score(self):
        score_display = self.font.render(str(self.score) + ' pr: ' + str(self.best_score), True, (255, 255, 255))
        self.screen.blit(score_display, (10, 10))

    def draw_generation(self):
        score_display = self.font.render('generation: ' + str(self.generation), True, (255, 255, 255))
        self.screen.blit(score_display, (800, 10))

    def draw_leading_birds_id(self):
        leading_birds_id = []
        count_alive_birds = 0
        for bird in self.birds:
            if bird['alive']:
                # append the bird id (index) to the list of leading birds id
                if len(leading_birds_id) < 5:
                    leading_birds_id.append(self.birds.index(bird))
                count_alive_birds += 1
        # display the id of the leading birds on the buttom left corner of the screen
        shift = 0
        for indx in leading_birds_id:
            leading_birds_id_display = self.font.render(str(indx), True, self.birds[indx]['color'])
            self.screen.blit(leading_birds_id_display, (10+shift, 550))
            shift += leading_birds_id_display.get_width() + 10

        #leading_birds_id_display = self.font.render(str(leading_birds_id), True, (255, 255, 255))
        #self.screen.blit(leading_birds_id_display, (10, 550))
        # display the number of birds that are still alive on the buttom right corner of the screen
        count_alive_birds_display = self.font.render(str(count_alive_birds), True, (255, 255, 255))
        self.screen.blit(count_alive_birds_display, (350, 550))

    def draw_game_over(self):
        end_text = self.font.render(f"Game Over! Score: {self.score}", True, (255, 255, 255))
        restart_text = self.font.render("Press 'R' to Restart", True, (255, 255, 255))
        self.screen.blit(end_text, (50, 250))
        self.screen.blit(restart_text, (50, 300))

    def generate_random_bird_color(self):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        return red, green, blue

    def generate_random_bird_radius(self):
        bird_radius = random.randint(15, 25)
        return bird_radius

    def play(self, path = None):
        self.init_game()
        best_neural_networks_indx = []
        while True:
            self.handle_events()
            if self.game_active:
                best_neural_networks_indx = self.update_game()
            else:
                self.init_game(best_neural_networks_indx)
            self.draw_game()
            pygame.display.update()
            self.clock.tick(120)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path=r'C:\Users\Vered\PycharmProjects\pythonProject1\SimpleNeuralNetwork.pkl'
    fb = fluppy_bird(file_path)
    fb.play()