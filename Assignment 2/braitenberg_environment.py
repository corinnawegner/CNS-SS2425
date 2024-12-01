import numpy as np
import matplotlib.pyplot as plt

class BraitenbergEnvironment:
    
    def __init__(self,  filename="parcour_1.png", 
                 vehicle_diameter=40., 
                 antenna_angle=30./180.*np.pi, 
                 initial_orientation=45./180*np.pi, 
                 short_antenna_length = 10., 
                 long_antenna_length=50.):
        """
        This class implements the whole vehicle within its environment.
        Use the `world_update` method to exchange motor commands and sensory
        readings with the vehicle and have the next time step be calculated.
    
        Environment configuration
            The world map file in `filename` should be a png with black obstacles.
            Best keep it black & white with white being free ground.
     
        Vehicle configuration
            You can specify the vehicle diameter, the angle in which the antennas
            stick out to the left and right. The most important experimental 
            parameters are the lengths of the short *reflex*- and long *predictor*-
            antennas.
                
            The vehicle is initialised in the centre of the map with orientation
            `initial_orientation`.
             
        Args:
            filename (str, optional): Defaults to "parcour_1.png".
            vehicle_diameter (float, optional): Defaults to 40.
            antenna_angle (float, optional): Defaults to 30./180.*np.pi.
            initial_orientation (float, optional): Defaults to 45./180*np.pi.
            short_antenna_length (float, optional): Defaults to 10. 
            long_antenna_length (float, optional): Defaults to 50. 
        """
        self.orientation      = initial_orientation
        self.diameter         = vehicle_diameter
        self.antenna_angle    = antenna_angle
        self.short_antenna    = short_antenna_length
        self.long_antenna     = long_antenna_length
        
        
        # Read PNG as black and white (1,0) matrix
        self.worldmap = 1.*( np.sum( plt.imread(filename)[:,:,:3], axis=2)<3).T
        self.worldmap  = np.flip(self.worldmap,axis=1)
        
        # Place vehicle at centre of map
        # We know it is empty there, because we made the image that way.
        self.X = np.shape(self.worldmap)[0]/2
        self.Y = np.shape(self.worldmap)[1]/2
        
        self.sensors= np.zeros(4)
        
    def read_antenna(self, angle, length):
        """
        Evaluate antenna in direction angle up to distance length.
        
        Check every map pixel in direction angle relative to vehicle heading.
        The check is extended up to length `length`. Returns free 
        `distance / length` in case of obstacle in range, otherwise returns 0.
        """
        
        antennax = np.ceil(self.X + np.arange(self.diameter,self.diameter+length, 0.01)* np.cos(self.orientation+angle)).astype(int)
        antennay = np.ceil(self.Y + np.arange(self.diameter,self.diameter+length, 0.01)* np.sin(self.orientation+angle)).astype(int)
                    
        i = np.where( 1 == self.worldmap[np.clip(antennax,0, self.worldmap.shape[0]-1),
                                         np.clip(antennay,0, self.worldmap.shape[1]-1)] )[0]
        
        if len(i):
            x_end = self.X+(self.diameter+length)*np.cos(self.orientation+angle)
            y_end = self.Y+(self.diameter+length)*np.sin(self.orientation+angle)
            return np.sqrt((x_end-antennax[i[0]])**2+(y_end-antennay[i[0]])**2)/length
        
        else:
            return 0   
        
    def world_update(self, motor_command):
        """
        Update world and return the sensory impression of the environment.
        
        Note:
            Unlike c++ template, the position and orientation of the vehicle are 
            also returned.
        """
        
        # The factor beta implements the low pass properties of antenna and world.
        beta = 0.9 
        
        # Limit maximum rotation speed to not turn more than pi/2.
        motor_command = np.clip(motor_command, -np.pi/2, +np.pi/2)
        
        # Update vehicle orientation
        self.orientation += motor_command
        
        # Vehicle geometry: both motors |v|=1 should result in a unit step;
        # |v| = 1 is the highest speed, except for rounding.
        # Note: One could check for collisions, but the original literature
        #       isn't checking and would complicate matters far more than
        #       a simple position check, because dead ends would be possible
        #       in front of obstacles.
        self.X += .3 * np.cos(self.orientation)  
        self.Y += .3 * np.sin(self.orientation) 
        
        # Update sensory values 
        #self.sensors *= beta 
        self.sensors[0] = (1. - beta) * self.read_antenna( self.antenna_angle, self.short_antenna) # short left
        self.sensors[1] = (1. - beta) * self.read_antenna(-self.antenna_angle, self.short_antenna) # short right
        self.sensors[2] = (1. - beta) * self.read_antenna( self.antenna_angle, self.long_antenna) # long left
        self.sensors[3] = (1. - beta) * self.read_antenna(-self.antenna_angle, self.long_antenna) # long right
        
        # uncomment to print out position and orientation of vehicle
        # print(self.X, self.Y, self.orientation, motor_command)
        
        return self.X, self.Y, self.orientation, self.sensors