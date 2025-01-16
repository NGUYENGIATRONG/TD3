import numpy as np
import matplotlib.pyplot as plt

# Load data from .npy files
roll_angles = np.load('roll_angles.npy')
pitch_angles = np.load('pitch_angles.npy')
yaw_angles = np.load('yaw_angles.npy')

# Create a sequence for time steps
steps = range(len(roll_angles))

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

# Plot roll angles
ax1.plot(steps, roll_angles, label='Roll Angle', color='blue')
ax1.set_ylabel('Roll (degrees)')
ax1.legend(loc='upper right')
ax1.grid(True)

# Plot pitch angles
ax2.plot(steps, pitch_angles, label='Pitch Angle', color='orange')
ax2.set_ylabel('Pitch (degrees)')
ax2.legend(loc='upper right')
ax2.grid(True)

# Plot yaw angles
ax3.plot(steps, yaw_angles, label='Yaw Angle', color='green')
ax3.set_ylabel('Yaw (degrees)')
ax3.set_xlabel('Time Steps')
ax3.legend(loc='upper right')
ax3.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()