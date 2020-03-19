# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:15:52 2019

@author: danie
"""

# Example of Kalman filter used for online estimation of ballistic missile
# trajectory.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
class Kalman_Filter:
    
    def __init__(self, A, B, M, H, Q, R):
        
        self.A = A
        self.B = B
        self.M = M
        self.H = H
        
        n,m = np.shape(A)
        self.n = n
        
        o,p = np.shape(H)
        self.o = o
        
        self.x = []
        #measurments
        self.z = []
        
        #noise covariances
        self.Q = Q #process covariance
        self.R = R #measurement covariance
        
        self.P_apri = []    #priori covariance estimate
        self.P_apost = []      #posteriori covariance estimate
        self.S = []    #output priori covariance estimate
        self.K = []        #Kalman gain
        
        
        self.xhat_apri = [] #a priori state estimate
        self.xhat_apost = [] #a posteriori state estimate
        
        #observation error
        self.ztilde = []
        
        
    def simulate(self, delT, t_max, x0, u, w, v, labels):
        self.delT = delT
        self.t_max = t_max
        self.t = np.arange(0,t_max,delT) #time vector
        
        self.x0 = x0
        self.x.append(self.x0)
        self.z.append(np.dot(self.H,self.x0))
        
        self.xhat_apri.append(x0)
        self.xhat_apost.append(x0)
        
        self.P_apost.append(np.zeros((self.n,self.n)))
        self.P_apri.append(np.zeros((self.n,self.n)))
        self.S.append(np.zeros((self.n,self.n)))
        self.K.append(np.zeros((self.n,self.n)))
        
        self.ztilde.append(np.zeros([1,self.o]))
        
        #control inputs and noise realizations
        self.u = u
        self.w = w
        self.v = v
        
        i = 0
        while(i < len(self.t) - 1):
            
            #dynamics
            self.x.append(self.A @ self.x[i] + self.B @ self.u[i,:] + self.M @ self.w[i,:])
            #measurment
            self.z.append(self.H @ self.x[i+1] + self.v[i+1,:])
            
            #Kalman Filter - Prediction Step
            self.P_apri.append(self.A @ self.P_apost[i] @ self.A.T + self.Q)
            self.xhat_apri.append(self.A @ self.xhat_apost[i] + self.B @ self.u[i,:])
            
            #Kalman Filter - Update Step
            self.ztilde.append(self.z[i+1] - self.H @ self.xhat_apri[i+1])
            
            self.S.append(self.H @ self.P_apri[i+1] @ self.H.T + R)

            self.K.append(self.P_apri[i+1] @ self.H.T @ np.linalg.inv(self.S[i+1]))
            
            self.xhat_apost.append(self.xhat_apri[i+1] + self.K[i+1] @ self.ztilde[i+1])
            
            P_gain = np.eye(self.n) - self.K[i+1] @ self.H

            self.P_apost.append(P_gain @ self.P_apri[i+1] @ P_gain.T + self.K[i+1] @ self.R @ self.K[i+1].T)
            
            #update iteration counter
            i += 1
            
        #store data in dataframe
        #convert data to dataframe
        self.df_x = pd.DataFrame(np.vstack(self.x), index = self.t, columns = labels[0])
        self.df_z = pd.DataFrame(np.vstack(self.z), index = self.t, columns = labels[1])
        self.df_xhat_apri = pd.DataFrame(np.vstack(self.xhat_apri), index = self.t, columns = labels[2])
        
        self.df = self.df_x.append([self.df_z, self.df_xhat_apri], sort=False)
        
        return self.df

##############################################################################

def setup_missile_dynamics(delT):

    #dynamics
    A = np.zeros((4,4))
    A[0,0] = 1
    A[0,1] = delT
    A[1,1] = 1
    A[2,2] = 1
    A[2,3] = delT
    A[3,3] = 1
    
    B = np.zeros((4,2))
    B[0,0] = 0.5*delT**2
    B[1,0] = delT
    B[2,1] = 0.5*delT**2
    B[3,1] = delT
    
    H = np.zeros((2,4))
    H[0,0] = 1
    H[1,2] = 1
    
    return A,B,H

def plot_results(df):

    plt.plot(df.index,df['x1'],label="x")
    plt.plot(df.index,df['z1'], label="z1")
    plt.plot(df.index,df['xhat1'], label="x hat")
    plt.legend()
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('X position')
    plt.show()
    #plt.savefig('x1_all.png')
    
    plt.plot(df.index,df['x2'], label="y")
    plt.plot(df.index,df['z2'], label="z2")
    plt.plot(df.index,df['xhat2'], label="y hat")
    plt.legend()
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('Y position')
    plt.show()
    #plt.savefig('x2_all.png')
    
    plt.plot(df.index,df['x1_dot'], label="x1_dot")
    plt.plot(df.index,df['xhat1_dot'], label="xhat1_dot")
    plt.legend()
    plt.ylabel('speed')
    plt.xlabel('time')
    plt.title('X velocity')
    plt.show()
    #plt.savefig('x1.png')
    
    plt.plot(df.index,df['x2_dot'], label="x2_dot")
    plt.plot(df.index,df['xhat2_dot'], label="xhat2_dot")
    plt.legend()
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('Y position')
    plt.show()
    #plt.savefig('x2.png')
    
##############################################################################
     
if __name__ == "__main__":
    
    print("Ballistic Missile Kalman Filter example:")
    
    #Simulation Parameters
    delT = 0.1    #timestep
    t_max = 50  #time simulation stops (seconds)
    t = np.arange(0, t_max, delT)
    
    g = -9.81   #m/s^2: gravity
    mass = 1  #kg
    
    #initial conditions
    x0 = np.zeros(4)
    
    init_angle = 75
    init_angle_radians = np.radians(init_angle)
    
    init_velocity = 500
    
    x0[0] = 0
    x0[1] = init_velocity*np.cos(init_angle_radians)
    x0[2] = 300 #initial height
    x0[3] = init_velocity*np.sin(init_angle_radians)
    
    #process noise: disturbance forces, normal distributions
    sig_px = 10
    sig_py = 1
    
    #measurement noise
    sig_mx = 200
    sig_my = 200
    
    Q = np.diag([sig_px,sig_px,sig_py,sig_py])
    R = np.diag([sig_mx,sig_my])
    
    A,B,H = setup_missile_dynamics(delT)
    
    #process and measurement noise
    w_x = np.random.normal(0,sig_px,(len(t),1))
    w_y = np.random.normal(0,sig_py,(len(t),1))
    w = np.hstack((w_x,w_y))
    
    v = np.hstack((np.random.normal(0,sig_mx,(len(t),1)), \
           np.random.normal(0,sig_my,(len(t),1))))
    
    #input (only input is acceleration due to gravity)
    u = np.hstack((np.zeros((len(t),1)), np.ones((len(t),1)) * g * mass))
    
    #setup Kalman Filter
    kalman = Kalman_Filter(A,B,B,H,Q,R)
    
    #dataframe column labels
    state_labels = ['x1','x1_dot','x2','x2_dot']
    measurement_labels = ['z1','z2']
    apriori_state_labels = ['xhat1','xhat1_dot','xhat2','xhat2_dot']
    labels = [state_labels, measurement_labels, apriori_state_labels ]
    
    #run the simulation
    df = kalman.simulate(delT, t_max, x0, u, w, v, labels)
    
    plot_results(df)
    
    print("Simulation Complete")
