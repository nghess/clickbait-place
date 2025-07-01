import matplotlib.pyplot as plt
import numpy as np

def generate_2d_normal_distribution(x_size=888, y_size=1968, mean_x=None, mean_y=None, 
                                  sigma_x=None, sigma_y=None, log_normal=False,
                                  log_sigma_x=0.5, log_sigma_y=0.5, visualize=True):
    """
    Generate a 2D normal or log-normal distribution.
    
    Parameters:
    -----------
    x_size : int
        Width of the output array (default: 888)
    y_size : int
        Height of the output array (default: 1968)
    mean_x : float, optional
        X-coordinate of the mean (default: center of x-axis)
    mean_y : float, optional
        Y-coordinate of the mean (default: center of y-axis)
    sigma_x : float, optional
        Standard deviation in x direction for normal dist (default: x_size/6)
    sigma_y : float, optional
        Standard deviation in y direction for normal dist (default: y_size/6)
    log_normal : bool, str, or list
        Control log-normal distribution:
        - False: normal distribution for both axes
        - True: log-normal for both axes  
        - 'x': log-normal for x-axis only
        - 'y': log-normal for y-axis only
        - ['x', 'y']: log-normal for both (same as True)
    log_sigma_x : float
        Standard deviation in log-space for x-axis log-normal (default: 0.5)
        Smaller values = tighter distribution, larger = fatter tails
    log_sigma_y : float
        Standard deviation in log-space for y-axis log-normal (default: 0.5)
        Smaller values = tighter distribution, larger = fatter tails
    visualize : bool
        Whether to plot the distribution (default: True)
        
    Returns:
    --------
    distribution : numpy.ndarray
        2D array containing the distribution
    """
    # Set default mean to center of array
    if mean_x is None:
        mean_x = x_size / 2
    if mean_y is None:
        mean_y = y_size / 2
        
    # Set default sigma proportional to array size
    if sigma_x is None:
        sigma_x = x_size / 6
    if sigma_y is None:
        sigma_y = y_size / 6
    
    # Parse log_normal parameter
    if log_normal is True:
        log_x, log_y = True, True
    elif log_normal is False:
        log_x, log_y = False, False
    elif log_normal == 'x':
        log_x, log_y = True, False
    elif log_normal == 'y':
        log_x, log_y = False, True
    elif isinstance(log_normal, (list, tuple)):
        log_x = 'x' in log_normal
        log_y = 'y' in log_normal
    else:
        raise ValueError("log_normal must be True, False, 'x', 'y', or a list containing 'x' and/or 'y'")
    
    # Create coordinate grids
    x = np.arange(x_size)
    y = np.arange(y_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distributions for each axis separately
    if log_x:
        # Log-normal for X axis
        log_mean_x = np.log(mean_x)
        Z_x = np.exp(-((np.log(np.maximum(X, 1)) - log_mean_x)**2 / (2 * log_sigma_x**2)))
    else:
        # Normal for X axis
        Z_x = np.exp(-((X - mean_x)**2 / (2 * sigma_x**2)))
    
    if log_y:
        # Log-normal for Y axis
        log_mean_y = np.log(mean_y)
        Z_y = np.exp(-((np.log(np.maximum(Y, 1)) - log_mean_y)**2 / (2 * log_sigma_y**2)))
    else:
        # Normal for Y axis
        Z_y = np.exp(-((Y - mean_y)**2 / (2 * sigma_y**2)))
    
    # Combine the distributions (multiply since they're independent)
    Z = Z_x * Z_y
    
    # Normalize to [0,1]
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    if visualize:
        plt.figure(figsize=(10, 10 * (y_size/x_size)))
        plt.imshow(Z, cmap='plasma')
        plt.axis('off')
        plt.show()
        
    return Z

# Log-normal on Y axis with fat tails
generate_2d_normal_distribution(log_normal='y', mean_y=400, log_sigma_y=.5)