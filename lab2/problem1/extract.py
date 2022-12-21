

import numpy as np
import torch
import matplotlib.pyplot as plt

file_path = r'sol\neural-network-1.pth'

try:
    model = torch.load(file_path).to('cpu')
    print('Loaded {}'.format(file_path))
    print('Network model: {}'.format(model))
except:
    print(f'{file_path} not found!')
    exit(-1)


n_points = 100  # Points per axis

y = np.linspace(0, 1.5, n_points)
w = np.linspace(-np.pi, np.pi, n_points)
Y, W = np.meshgrid(y, w)

states = np.zeros((n_points**2, 8))
states[:, 1] = Y.flatten()
states[:, 4] = W.flatten()

Q = model(torch.tensor(states, dtype=torch.float32)) # Extract desired Q values
max_values, max_indices  = torch.max(Q, 1)
Q_max = max_values.detach().numpy().reshape(n_points, n_points)
Q_argmax = max_indices.detach().numpy().reshape(n_points, n_points)

cmap_type = 'viridis'
fig = plt.figure()
fig.set_size_inches(12, 6)
fig.suptitle('{} points'.format(n_points**2))

ax = fig.add_subplot(121, projection='3d')
ax.set_title('Maximum Q values')
ax.plot_surface(Y, W, Q_max, cmap=cmap_type)
ax.contour(Y, W, Q_max, zdir='x', offset=-0.1, levels=3, cmap=cmap_type)
ax.contour(Y, W, Q_max, zdir='y', offset=3.4, levels=3, cmap=cmap_type)
ax.set_xlabel('Altitude (y)')
ax.set_ylabel('Angle ($\omega$)')
ax.set_zlabel('Maximum Q value')

ax = fig.add_subplot(122, projection='3d')
ax.scatter(Y, W, Q_argmax, s=1, c=Q_argmax, cmap=cmap_type)
ax.set_title('Best actions')
ax.set_xlabel('Altitude (y)')
ax.set_ylabel('Angle ($\omega$)')
ax.set_zlabel('Best action')
ax.set_zticks([0, 1, 2, 3])
ax.set_zticklabels(['Nothing', 'Left', 'Main', 'Right'])

plt.show()


