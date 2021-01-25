import numpy as np 
import matplotlib.pyplot as plt 

Kp = 1.18
Ti = 0.13
Td = 0.15

class PID():
    def __init__(self, Kp=Kp, Ti=Ti, Td=Td):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.previous_error = 0.0
        self.previous_integral = 0.0
        self.integral = 0.0
        self.threshold = np.deg2rad(30)
        self.threshold_speed = np.deg2rad(10)
        self.pervious_delta = 0.0


    def control(self, desired_val, curent_val, time_delta):
        
        error = desired_val - curent_val #cal error
        previous_error = self.previous_error
        self.previous_error = error
        self.previous_integral = self.integral

        derivative = (error - previous_error) / time_delta
        self.integral += error * time_delta

        delta = - self.Kp*(error + derivative*self.Td + self.integral/self.Ti)
        
        #constrain detla speed
        if (np.abs(delta - self.pervious_delta) > self.threshold_speed).all():
            if delta - self.pervious_delta > self.threshold_speed:
                delta = self.pervious_delta + self.threshold_speed
            elif delta - self.pervious_delta < self.threshold_speed:
                delta = self.pervious_delta - self.threshold_speed
        
        #constrain detla
        if delta > self.threshold:
            delta = self.threshold
            self.integral = self.previous_integral # reset integral

        elif delta < -self.threshold:
            delta = -self.threshold
            self.integral = self.previous_integral # reset integral

        self.pervious_delta = delta
        
        return delta
