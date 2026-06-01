import numpy as np

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