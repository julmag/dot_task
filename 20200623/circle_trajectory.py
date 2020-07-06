import math
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera


def circle (fixed_params=False, simple_circle = True):
    rmin = 0.01
    rmax = 0.3
   
    if fixed_params:
        center_x = 0
        center_y = 0.35

        ellipse_rotation = 0
        grad_start = 0
        delta_grad = 2*np.pi/1080

        a = 0.1
        b = 0.1
   
    else:
        x_range = -0.1, 0.1
        y_range = 0.3, 0.4

        delta_g_range = 2*np.pi/1080, 2*np.pi/360 # 1/1080 circle (= 0.33°) to 1/360 circle (= 1°) per step
    
    
    
    center_x = np.random.uniform(x_range[0], x_range[1]) if not fixed_params else center_x
    center_y = np.random.uniform(y_range[0], y_range[1]) if not fixed_params else center_y

    rmax_a = min(1-(center_x**2+center_y**2)**0.5, rmax)

    if rmax_a > rmin:

        if simple_circle:
            ellipse_rotation = 0
        else:
            ellipse_rotation = np.random.uniform(0, np.pi*0.5) if not fixed_params else ellipse_rotation

        grad_start = np.random.uniform(0, np.pi*2) if not fixed_params else grad_start
        delta_grad = np.random.uniform(delta_g_range[0], delta_g_range[1]) * np.random.choice([1,-1]) if not fixed_params else delta_grad
        grad = grad_start

        a = np.random.uniform(rmin, rmax_a)
        # ensuring that a and b are of the same magnitude
        if simple_circle:
            b = a
        elif fixed_params==False:
            rmin_b = max(rmin, 0.5*a)
            rmax_b = min(rmax_a, 2*a)
            b = np.random.uniform(rmin_b, rmax_b)

        # ensuring the ellipse doesn't touch the arm base
        while np.abs(a - (center_x**2+center_y**2)**0.5) < 0.05 or np.abs(b - (center_x**2+center_y**2)**0.5) < 0.05:
            a = np.random.uniform(rmin, rmax)
            # ensuring that a and b are of the same magnitude
            if simple_circle:
                b = a
            else:
                rmin_b = max(rmin, 0.5*a)
                rmax_b = min(rmax, 2*a)
                b = np.random.uniform(rmin,rmax)

        if simple_circle:
            data = []
            data = center_x, center_y, a, grad_start
        else:
            data = []
            data = center_x, center_y, a, b, grad_start
           
    return data


def circle_trajectory(armlength, steps, center, r, rotation=0, grad_start=0, delta_grad=None, debug=False, make_gif=False):
    """
        Computes points along a circle / ellipsis and corresponding joint angle updates
        of 2 DOF arm for moving its tip along trajectory (assumed base location (0,0))

        Parameters:
        ----------
            * armlength: length of arm elements, tuple or single value if both are same length
            * steps: number of time steps along trajectory
            * center: center of circle / ellipsis, tuple
            * r: radius of circle / ellipsis, single or tuple respectively
            * rotation: rotation of ellipsis [0, 2*pi], default 0
            * grad_start: defines starting position on circle / ellipsis [0, 2*pi], default 0
            * delta_grad: length of steps along trajectory [0, 2*pi], default is 2*pi/steps
            * debug: plot arm positions

        Returns
        -------
            Array (steps x 6)
            * x and y position along trajectory
            * "shoulder" and "elbow" angle deltas
            * x and y position along trajectory for the next step

    """
    data = np.zeros((steps+1, 6))

    grad = grad_start
    if isinstance(r, (float, int)):
        r = (r, r)
    if isinstance(armlength, (float, int)):
        armlength = (armlength, armlength)

    # full circle within range of steps
    if delta_grad is None:
        delta_grad = 2*np.pi / steps

    for i in range(steps+1):

        # no increment in first step
        if i:
            grad += delta_grad
            grad %= 2*np.pi

        # compute position on trajectory
        x_uncentered = r[0] * math.cos(grad)
        y_uncentered = r[1] * math.sin(grad)

        x_rot = x_uncentered * math.cos(rotation) - y_uncentered * math.sin(rotation)
        y_rot = x_uncentered * math.sin(rotation) + y_uncentered * math.cos(rotation)

        x = x_rot + center[0]
        y = y_rot + center[1]

        # compute arm coordinates
        dist_base = (x**2 + y**2)**0.5
        theta_base = math.atan2(y, x)

        theta_elbow = math.acos((armlength[0]**2 + armlength[1]**2 - dist_base**2) / (2*armlength[0] * armlength[1]))
        theta_shoulder = math.acos((armlength[1]**2 + dist_base**2 - armlength[0]**2) / (2*dist_base * armlength[1])) + theta_base

        if debug:
        # checking that arm elements have correct length and plot arm 
            x_elbow = armlength[0] * math.cos(theta_shoulder)
            y_elbow = armlength[0] * math.sin(theta_shoulder)
            upper_arm = (x_elbow**2 + y_elbow**2)**0.5
            lower_arm = ((x - x_elbow)**2 + (y - y_elbow)**2)**0.5
            if np.abs(upper_arm - armlength[0]) > 0.001:
                print("[Warning] Upper arm wrong length", upper_arm)
            if np.abs(lower_arm - armlength[1]) > 0.001:
                print("[Warning] Lower arm wrong length", lower_arm)
            
            plt.scatter(0,0)
            plt.scatter(x_elbow, y_elbow)
            plt.scatter(x, y)
            plt.scatter(data[:i, 0], data[:i, 1], c='black', s=1)
            plt.plot([0, x_elbow],[0, y_elbow])
            plt.plot([x_elbow, x],[y_elbow, y])
            plt.ylim([-sum(armlength), sum(armlength)])
            plt.xlim([-sum(armlength), sum(armlength)])
            plt.show()
            if i > 0:
                if np.abs(theta_shoulder - data[i-1, 3]) > delta_grad + 0.001:
                    print("[Warning] Shoulder angle jump", theta_shoulder - data[i-1, 3], 
                          "From", data[i-1, 3], "to", theta_shoulder)
                if np.abs(theta_elbow - data[i-1, 2]) > delta_grad + 0.001:
                    print("[Warning] Elbow angle jump", theta_elbow - data[i-1, 2],
                          "From", data[i-1, 2], "to", theta_elbow)
                

            
        # save
        data[i, 0:4] = x, y, theta_elbow, theta_shoulder
        if i > 0:
            data[i-1, 2:4] = theta_elbow - data[i-1, 2], theta_shoulder - data[i-1, 3]
            data[i-1, 4:6] = x, y
            

    return data[:-1, :]



