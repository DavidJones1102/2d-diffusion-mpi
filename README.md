# 2d-diffusion-mpi

## 1. Problem Description
This project implements a parallel numerical solver for the two-dimensional diffusion equation using mpi4py. \
The equation is given by:

$$
\frac{\partial P(x,y,t)}{\partial t} = \frac{\partial^2 P(x,y,t)}{\partial x^2} + \frac{\partial^2 P(x,y,t)}{\partial y^2}
$$

Where:
* $P(x,y,t)$ represents density of the substance at position $(x,y)$ and time $t$.

## 2. Problem discretization
To solve this equation, we employ the **Finite Difference Method**. We discretize the continuous domain into a grid.


### Step A: time derivative approximation
$$
\frac{\partial P}{\partial t} \approx \frac{P_{i,j}^{(n+1)} - P_{i,j}^{(n)}}{\Delta t}
$$

### Step B: spatial derivatives approximation

$$
\frac{\partial^2 P}{\partial x^2} \approx \frac{P_{i+1,j}^{(n)} - 2P_{i,j}^{(n)} + P_{i-1,j}^{(n)}}{(\Delta x)^2}
$$

$$
\frac{\partial^2 P}{\partial y^2} \approx \frac{P_{i,j+1}^{(n)} - 2P_{i,j}^{(n)} + P_{i,j-1}^{(n)}}{(\Delta y)^2}
$$

### Step C: formulate discrete equation

$$
\frac{P_{i,j}^{(n+1)} - P_{i,j}^{(n)}}{\Delta t} = \left[ \frac{P_{i+1,j}^{(n)} - 2P_{i,j}^{(n)} + P_{i-1,j}^{(n)}}{(\Delta x)^2} + \frac{P_{i,j+1}^{(n)} - 2P_{i,j}^{(n)} + P_{i,j-1}^{(n)}}{(\Delta y)^2} \right]
$$

Rearranging to solve for the future state $P_{i,j}^{(n+1)}$:

$$
P_{i,j}^{(n+1)} = P_{i,j}^{(n)} + \Delta t \left[ \frac{P_{i+1,j}^{(n)} - 2P_{i,j}^{(n)} + P_{i-1,j}^{(n)}}{(\Delta x)^2} + \frac{P_{i,j+1}^{(n)} - 2P_{i,j}^{(n)} + P_{i,j-1}^{(n)}}{(\Delta y)^2} \right]
$$

## 3. Numerical stability
This explicit scheme is conditionally stable. To prevent the simulation from blowing up, the time step $\Delta t$ must be sufficiently small. The stability limit is given by:

$$
\Delta t \le \frac{(\Delta x \Delta y)^2}{2(\Delta x)^2 + 2(\Delta y)^2}
$$

## PCAM analysis
| $y \setminus x$ | $i-1$ | $i$ | $i+1$ |
| :---: | :---: | :---: | :---: |
| **$j+1$** | | $P_{i,j+1}$ | |
| **$j$** | $P_{i-1,j}$ | $\mathbf{P_{i,j}}$ | $P_{i+1,j}$ |
| **$j-1$** | | $P_{i,j-1}$ | |

### 1. Partitioning
- **Task**: Each grid point $P_{i,j}$ represents one independent task.
- **Result**: $N_x \times N_y$ primitive tasks.

### 2. Communication
To calculate $P_{i,j}^{(n+1)}$, we need $P_{i+1,j}^{(n)}$, $P_{i-1,j}^{(n)}$, $P_{i,j+1}^{(n)}$, and $P_{i,j-1}^{(n)}$.

### 3. Agglomeration
We group primitive tasks into larger chunks to match the available hardware. We use row decomposition, where contiguous rows are bundled together and assigned to a specific MPI process.

| $y \setminus x$ | 0 | 1 | 2 | 3 | Process |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `-` | `-` | `-` | `-` | Rank 0 |
| 1 | `-` | `-` | `-` | `-` | Rank 0 |
| 2 | `+` | `+` | `+` | `+` | Rank 1 |
| 3 | `+` | `+` | `+` | `+` | Rank 1 |
| 4 | `)` | `)` | `)` | `)` | Rank 2 |
| 5 | `)` | `)` | `)` | `)` | Rank 2 |

### 4. Mapping
We map each agglomerated block directly to a physical CPU core, assigning ranks linearly to ensure that processes handling adjacent rows remain logically close to minimize communication latency.


