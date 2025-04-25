import numpy as np

def initial_prior(hall):
    """
    Initialize the probability distribution
    hall: a numpy array containing the features at each position
    Returns an array containing the initialized probability distribution
    """
    return np.full(hall.size, 1 / hall.size)

def normalize(v):
    """
    Make the values in the array a valid probability distribution
    by scaling them so that they sum to 1.
    v: the array to normalize
    Returns a normalized array
    """
    sum = np.sum(v)
    if sum == 0:
        return v 
    else:
        return v / sum

def update(hall, prior, displacement, observation):
    """
    Update the prior based on new position and measurement
    hall: a numpy array containing the features at each position
    prior: the prior probability distribution
    displacement: offset from the dog's previous position
    observation: feature reported by the sensor
    Returns the posterior probability distribution
    """ 

    # These probabilities assume 100% accurate sensors
    # Probability of a door given the hall structure
    p_door = 1 / np.sum(hall)
    # Probability of a wall given the hall structure
    p_wall = 1 / (hall.size - np.sum(hall))

    # Shift the prior to take the dog's movement into account.
    # For example, if there was a 50% chance the dog was at
    # position 3, then when it moves right, that 50% chance is
    # now assigned to the position to the right, i.e. position 4.
    prior = np.roll(prior, shift=displacement)
    print("Prior      ", end="")
    print(prior)

    # Determine probability distribution for the sensor measurement.
    if observation == 1:
        # Every position with a door has probabilty p_door
        # and every position with a wall has 0 probability
        cp = hall * p_door
    else:
        # Every position with a wall has probability p_wall
        # and every position with a door has 0 probability
        cp = np.logical_not(hall) * p_wall

    # Update probability distribution by multiplying prior and measurement likelihood
    # and normalizing the result
    posterior = normalize(prior * cp)

    print("Posterior  ", end="")
    print(posterior)
    print("------------------------------------------------------------------------")
    return posterior


# Set print options to make a compact display
np.set_printoptions(precision=2, suppress=True, threshold=100, edgeitems=2, linewidth=200,
                    formatter={'float': '{:5.2f}'.format, 'int': '{:5d}'.format})

# 1=Door, 0=Wall
hall = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
print("Hall       ", end="")
print(hall)

# Initial prior
pd = initial_prior(hall)
print("------------------------------------------------------------------------")

# Updates
pd = update(hall, pd, 0, 0)  # Observe wall at initial position
pd = update(hall, pd, 1, 1)  # Move right, observe door
pd = update(hall, pd, 2, 1)  # Move right, observe door
