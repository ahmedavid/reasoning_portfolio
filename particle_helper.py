import numpy as np

def simulate_ball_trajectory(initial_positions, launch_speeds, launch_angles, time_step=0.01, total_time=2):
    """
    Simulates the trajectory of a ball launched from a given height with a specified speed and angle.

    Parameters:
        initial_positions (np.array): The initial position from which the ball is launched (in meters).
        launch_speeds (np.array): The speed at which the ball is launched (in meters per second).
        launch_angles (np.array): The angle at which the ball is launched (in degrees).
        time_step (float): The time interval between calculations (in seconds). Default is 0.1 seconds.
        total_time (float): The total duration for which the simulation runs (in seconds). Default is 100 seconds.

    Returns:
        An array of arrays where each one contains the x and y coordinates of the ball's position at each time step.
    """
    # Define the acceleration due to gravity (m/s^2)
    g = 9.81
    
    # Generate an array of time points from 0 to total_time with intervals of time_step
    time_points = np.arange(0, total_time, time_step)
    
    positions1 = []
    
    # Loop over each time point to calculate the ball's position and velocity
    for t in time_points:
        # Ball 1
        launch_angle_rad1 = np.radians(launch_angles[0])
        vx1 = launch_speeds[0] * np.cos(launch_angle_rad1)
        vy1 = launch_speeds[0] * np.sin(launch_angle_rad1)
        x1 = initial_positions[0][0] + vx1 * t
        y1 = initial_positions[0][1] + vy1 * t - 0.5 * g * t**2
        if y1 >= 0:
            positions1.append((x1, y1))
        
    return np.array(positions1)

def simulate_observations(true_positions, observation_noise_std, num_drop_out_interval=0):
    """
    Simulates noisy observations of true positions and introduces missing data at specified intervals.

    Parameters:
        true_positions (numpy.ndarray): The true positions of the object (array of shape (n, 2) for 2D positions).
        observation_noise_std (float): The standard deviation of the Gaussian noise added to the true positions.
        num_drop_out_interval (int): The interval number at which the data will be replaced with NaNs. Default is 0 (no dropout).

    Returns:
        An array of positions with added Gaussian noise and NaNs introduced at the specified interval.
    """    
    # Add Gaussian noise to the true positions
    noisy_positions = true_positions + np.random.normal(0, observation_noise_std, true_positions.shape)
    
    # Calculate the range for the intervals where data will be replaced with NaNs
    intervals_range = len(noisy_positions) // 5
    
    f = (num_drop_out_interval - 1) * intervals_range   # Start index for the dropout interval
    l = num_drop_out_interval * intervals_range         # End index for the dropout interval

    # Replace the specified interval with NaNs to simulate dropout
    noisy_positions[f: l] = np.nan

    # Return the noisy positions with the introduced NaNs
    return noisy_positions

class ParticleFilter:
    """
    A class to implement a Particle Filter for tracking an object's position and velocity.

    Attributes:
        num_particles (int): Number of particles in the filter.
        process_noise_std (float): Standard deviation of the process noise.
        observation_noise_std (float): Standard deviation of the observation noise.
        particles (numpy.ndarray): Array containing the state of each particle (shape: (num_particles, 4)).
        weights (numpy.ndarray): Array containing the weights of each particle (shape: (num_particles,)).
    """

    def __init__(self, num_particles, process_noise_std, observation_noise_std, initial_positions):
        """
        Initializes the ParticleFilter with the given parameters.

        Parameters:
            num_particles (int): Number of particles to use in the filter.
            process_noise_std (float): Standard deviation of the process noise.
            observation_noise_std (float): Standard deviation of the observation noise.
            initial_positions (numpy.ndarray): Initial positions and velocities of the particles.
        """
        self.num_particles = num_particles
        self.process_noise_std = process_noise_std
        self.observation_noise_std = observation_noise_std
        self.particles = np.zeros((num_particles, 4))  # [x, y, vx, vy] for each particle
        self.weights = np.ones(num_particles) / num_particles
        self.particles = np.tile(initial_positions, (num_particles, 1)) + np.random.randn(num_particles, 4) * observation_noise_std
        
    def predict(self, dt):
        """
        Predicts the next state of the particles based on the elapsed time.

        Parameters:
            dt (float): The time interval over which to predict the particle states.
        """
        g = 9.81
        # Update positions and velocities with process noise
        self.particles[:, 0] += self.particles[:, 2] * dt + np.random.normal(0, self.process_noise_std, self.num_particles)
        self.particles[:, 1] += self.particles[:, 3] * dt - 0.5 * g * dt**2 + np.random.normal(0, self.process_noise_std, self.num_particles)
        self.particles[:, 2] += np.random.normal(0, self.process_noise_std, self.num_particles)
        self.particles[:, 3] -= g * dt + np.random.normal(0, self.process_noise_std, self.num_particles)

    def update(self, observations):
        """
        Updates the particle weights based on the observed positions.

        Parameters:
            observations (numpy.ndarray): The observed positions.
        """
        observations = np.array(observations)
        new_p = self.particles[:, :2] + np.random.normal(0, self.process_noise_std, 2 * self.num_particles).reshape(self.num_particles, 2)
        
        if not np.isnan(observations).any():
            distances1 = np.linalg.norm(new_p - observations.reshape(1, 2), axis=1)
            self.weights = np.exp(-0.5 * (distances1**2) / self.observation_noise_std**2)
            self.weights += 1.e-300  # to avoid round-off to zero
            self.weights /= sum(self.weights)

    def resample(self):
        """
        Resamples the particles based on their weights to focus on the more likely particles.
        """
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # to avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.random(self.num_particles))
        
        self.particles = self.particles[indexes, :]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """
        Estimates the current state based on the weighted mean of the particles.

        Returns:
            numpy.ndarray: The estimated state.
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)
        return mean

    