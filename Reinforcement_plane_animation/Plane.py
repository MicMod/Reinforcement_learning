import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

class Plane():
    
    def __init__(self,data_path="/Data/Mp(10alpha).csv", reward_type='L1+Gauss', evaluate=False):
        #Plane data
        self.Jyy = 0.02478 # moment of inertia
        self.ro = 1.225 # density [kg/m^3]
        self.S = 0.16321 # wings area [m^2]
        self.ca = 0.162 # mean aerodynamic chord [m] 
        self.kh = 0.522 # ratio of horizontal stabilizer
        self.a1 = 3.33 # delta_Cl/delta_alpha [1/rad]
        self.V = 10 # airspeed
        self.Ms = self.read_data(file_path=data_path, col_name='alpha' ) # pitching moment (wings + stabilizer)
        # Angles and constrains
        self.delta = 0.0 # angle of all-moving tailplane
        self.theta = 0.0 # pitch angle 
        self.setpoint = np.deg2rad(5)
        self.delta_range = np.deg2rad(30) 
        self.threshold_theta_speed = np.deg2rad(10)
        self.min_theta_threshold = np.deg2rad(-25.0)
        self.max_theta_threshold = np.deg2rad(25.0)
        self.theta_range= self.max_theta_threshold-self.min_theta_threshold
        
        self.state = (0.0, 0.0, 0.0, 0.0)
        self.tau = 0.02 # seconds between updates - frequency 50 Hz
        self.reward_type = reward_type
        self.evaluate = evaluate
        self.sigma = 0.5 # standard deviation in reward function "L1+Gauss"
        
        
    def read_data(self, file_path, col_name):
        data = pd.read_csv(file_path)
        data.set_index([col_name], inplace=True)
        return data
    
    # calculate pitching moment produced by deflection of all-moving tailplane
    def cal_Mh(self, delta):
        return 0.5 * self.ro * self.V**2 * self.S * self.ca * self.kh * self.a1 * delta
        
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint
    
    
    def step (self, action):
        
        theta, theta_vel, delta , error = self.state
        previous_theta = delta
        delta = np.float64(action)
        
        #constrain theta
        if (np.abs(delta - previous_theta) > self.threshold_theta_speed):
            
            if ((delta - previous_theta) > self.threshold_theta_speed):
                delta = previous_theta + self.threshold_theta_speed
                
            elif ((delta - previous_theta) < self.threshold_theta_speed):
                delta = previous_theta - self.threshold_theta_speed

        theta_deg = round(np.rad2deg(theta),1)
        int_theta_deg = int(10*theta_deg)
        
        #cal theta
        theta_acc = (self.Ms.at[int(int_theta_deg), 'Ms'] - self.cal_Mh(delta))/self.Jyy
        theta_vel += theta_acc * self.tau
        theta +=  theta_vel * self.tau
        
        error = self.setpoint - theta
        
        self.state =  (theta, theta_vel, delta, error)
        
        
        return np.array(self.state), self.reward(), self.done(), {}
    
    #reset states and setpoint
    def reset(self):            
        state_= []  
        if self.evaluate:
            state_[:2] = [0.0, 0.0, 0.0]#np.deg2rad(5)*(2*np.random.rand(3).astype(np.float16)-1) # random theta, theta_vel, delta in range(-5; 5) deg
        else:
            state_[:2] = np.deg2rad(5)*(2*np.random.rand(3).astype(np.float16)-1) # random theta, theta_vel, delta in range(-5; 5) deg
        state_.append(self.setpoint - state_[0]) # error
        self.state = tuple(state_)
        #reset setpoints
        self.set_setpoint(np.deg2rad(np.random.randint(-10,10)))
        
        return np.array(self.state)
    
    def reward(self):
        if(self.reward_type == 'L1'):
            return -abs(np.rad2deg(self.setpoint - self.state[0])) # ujemne nagrody - 0 gdy perfekcyjnie
        elif(self.reward_type == 'L1+Gauss'):
            L1 = -abs(np.rad2deg(self.setpoint - self.state[0]))
            gauss = 1/(self.sigma*np.sqrt(2*3.14))*np.exp(-(np.rad2deg(self.state[0]-self.setpoint))**2/(2*self.sigma**2))
            return  L1+gauss
        
    def done(self):
        theta = self.state[0]
        if  (theta > self.max_theta_threshold) or (theta < self.min_theta_threshold):
            return True
        else:
            return False
        
