import pygame, sys, os
import pandas as pd
import numpy as np
from pygame.math import Vector2
import glob
from PIL import Image
import time
from Plane import Plane

class  Part():
    
    def __init__(self, scale=400, start_pos=Vector2(300,400), points_file='Data/naca0012.csv', SA=0):
        self.scale = scale # to scale points
        self.start_pos = start_pos 
        self.points_file = points_file
        self.points = self.read_Part_points() 
        self.CoR = Vector2(self.start_pos.x + SA,  self.start_pos.y) # CoR - Center of Rotation
        self.angle_vel = 0.0
        
    def read_Part_points(self):
        points = []
        data = pd.read_csv(self.points_file)
        X = data['X'].to_numpy()
        Y = data['Y'].to_numpy()
        points = [Vector2(self.start_pos.x + x*self.scale, self.start_pos.y + y*self.scale) for x, y in zip(X, Y)]
        
        return points
        
    def rotate(self, CoR, angle_vel):
        
        self.points = [Vector2(CoR.x+(p.x-CoR.x)*np.cos(angle_vel)-(p.y-CoR.y)*np.sin(angle_vel), CoR.y+(p.x-CoR.x)*np.sin(angle_vel)+(p.y-CoR.y)*np.cos(angle_vel)) for p in self.points]   
        
    def rotate_CoR(self, CoR, angle_vel):
        
        self.CoR = Vector2(CoR.x+(self.CoR.x-CoR.x)*np.cos(angle_vel)-(self.CoR.y-CoR.y)*np.sin(angle_vel), CoR.y+(self.CoR.x-CoR.x)*np.sin(angle_vel)+(self.CoR.y-CoR.y)*np.cos(angle_vel)) 
        
    
def read_obs(obs_path='Data/states.csv'):
    obs = pd.read_csv(obs_path)
    obs = np.deg2rad(obs.to_numpy())
    return obs

def draw_neural_net(window):
    nn_id = np.random.randint(low=1, high=8)
    image = pygame.image.load(r'Data/NN/nn{}.png'.format(nn_id))
    window.blit(image, (-140, -213)) 
    
def draw_rect(window):
    #up
    pygame.draw.rect(window, (0,0,0), rect=(330,40, 650, 250), width=2)
    #down
    pygame.draw.rect(window, (0,0,0), rect=(330,480, 650, 220), width=2)
    
def draw_lines(window):
    #right
    pygame.draw.lines(window, (0,0,0), closed=False, points=[(980,165),(1100,165), (1100, 590), (980, 590)], width=2)
    pygame.draw.polygon(window, (0,0,0), points=[(980, 590), (995, 580), (995, 600)])
    #center
    pygame.draw.lines(window, (0,0,0), closed=False, points=[(655,290),(655,480)], width=2)
    pygame.draw.polygon(window, (0,0,0), points=[(655,290), (645,305), (665,305)])
    #left
    pygame.draw.lines(window, (0,0,0), closed=False, points=[(330,165),(210,165), (210, 590), (330, 590)], width=2)
    pygame.draw.polygon(window, (0,0,0), points=[(330,165), (315,155), (315,175)])

def draw_net_states(window, font, data):

    for i in range(len(data)):
        
        text = font.render('{}'.format(data[i]), False, (0, 0, 0))
        window.blit(text, (365, 125+i*24))

def draw_net_action(window, font, data):
    
    text = font.render('{}'.format(data), False, (0, 0, 0))
    window.blit(text, (925, 160))
    

def display():
    screen = pygame.display.set_mode((1280, 720))
    
    #Parts of aircraft
    plane = Part(scale=1.2, start_pos=Vector2(600,600),  points_file='Data/aircraft_coordinates_transformed.csv')
    tail_wing = Part(scale=90, start_pos=Vector2(800,595), points_file='Data/naca0012.csv', SA=90*0.25)
    axi_xx = Part(scale=300, start_pos=Vector2(600,600),  points_file='Data/line.csv')
    
    #Fonts
    bigfont = pygame.font.SysFont('Arial', 21)
    smallfont = pygame.font.SysFont('Arial', 10)
    
    #Time
    fps = 500
    clock = pygame.time.Clock()
    delta_time = 0.0
    dt = 0.02
    index = 0
    
    #States-observations
    old_plane_theta = 0.0
    old_tail_theta = 0.0
    observations = read_obs()
    theta, theta_dot, delta, erorr = zip(*observations) 
    obs_to_dispaly = np.around(np.rad2deg(np.rad2deg(observations)),decimals=1)
    obs_to_dispaly = np.vstack(([[0.0, 0.0, 0.0, 10]], obs_to_dispaly))
    
    #Video
    file_num = 0
    try:
        os.makedirs("Snaps")
    except OSError:
        pass
    
    #Main loop
    while True:
        # Exit 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)
        
        #Cal time    
        delta_time += clock.tick()
 
        while delta_time > fps:
            screen.fill((255,255,255))
            
            #Draw text
            text_step = bigfont.render('Step {}'.format(index+1), False, (0, 0, 0))
            text_action = bigfont.render('Action: {}'.format(obs_to_dispaly[index+1][2]), False, (0,0,0))
            text_reward = bigfont.render('Reward: {}'.format(-abs(obs_to_dispaly[index+1][3])), False, (0,0,0))
            text_state = bigfont.render('States:', False, (0,0,0))
            text_state_val = bigfont.render('{}'.format(obs_to_dispaly[index+1]), False, (0,0,0))
            screen.blit(text_step, (50, 50))
            screen.blit(text_action, (1120, 365))
            screen.blit(text_reward, (665, 365))
            screen.blit(text_state, (70,350))
            screen.blit(text_state_val, (8,375))
            
            #Draw frames and lines
            draw_rect(window=screen)
            draw_lines(window=screen)

            #Draw neural net
            draw_neural_net(window=screen)
            draw_net_states(window=screen, font=smallfont, data=obs_to_dispaly[index])
            draw_net_action(window=screen, font= smallfont, data=obs_to_dispaly[index+1][2])
        
            ###Update possition of the plane
            ##Cal new angular velocity 
            #Plane
            plane.angle_vel = (theta[index]-old_plane_theta)/dt
            old_plane_theta = theta[index]
            #Tail wing
            tail_wing.angle_vel = (delta[index]-old_tail_theta)/dt
            old_tail_theta = delta[index]
            ##Rotate parts
            plane.rotate(CoR=plane.CoR, angle_vel=plane.angle_vel)
            tail_wing.rotate(CoR=plane.CoR, angle_vel=plane.angle_vel)
            tail_wing.rotate_CoR(CoR=plane.CoR, angle_vel=plane.angle_vel)
            tail_wing.rotate(CoR=tail_wing.CoR, angle_vel=tail_wing.angle_vel)
            axi_xx.rotate(CoR=plane.CoR, angle_vel=plane.angle_vel)
            
            ##Draw
            #Draw setpoint line
            SP_deg = 9
            pygame.draw.line(screen, (255,0,0),((plane.start_pos[0]+385*(np.cos(np.deg2rad(SP_deg)))), (plane.start_pos[1]+385*(np.sin(np.deg2rad(SP_deg))))), ((plane.start_pos[0]-275*(np.cos(np.deg2rad(SP_deg)))), (plane.start_pos[1]-275*(np.sin(np.deg2rad(SP_deg))))), width=2)
            #Draw parts of plane
            pygame.draw.polygon(screen, (105,105,105), plane.points, width =0)
            pygame.draw.polygon(screen, (255,150, 0), tail_wing.points)
            pygame.draw.lines(screen, (0,0,0),closed=False, points=axi_xx.points, width=2)
        
            #Increment values 
            index +=1
            file_num = file_num + 1
            
            # Save every frame
            filename = "Snaps/%04d.png" % file_num
            pygame.image.save(screen, filename)

            #Reset delta_time
            delta_time -= 2000
            pygame.display.flip() 
            


def make_gif():
    # filepaths
    fp_in = "Snaps/*.png"
    fp_out = "Snaps/nn_plane.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)




if __name__ == "__main__":
    
    pygame.init()
    
    display()
    
    #make_gif()