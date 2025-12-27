#!/usr/bin/env python
import argparse
import numpy as np
from mpi4py import MPI
import os
import csv

def get_index(x, y, N):
    return y*N + x

def get_value(matrix, upper_row, lower_row, x, y, N):
    # Boundary condition on x
    if x < 0 or x >= N:
        return 0.0
    rows_in_chunk = matrix.shape[0] // N
    
    if y < 0:
        return upper_row[x]
    if y >= rows_in_chunk:
        return lower_row[x]
    
    index = get_index(x, y, N)
    return matrix[index]

def step(local_data, start_row, end_row, N, dx, dy, dt):
    old = local_data.copy()
    local_height = end_row - start_row
    
    reqs = []
    # Boundary condition on y
    upper_row = np.zeros(N, dtype=old.dtype)
    lower_row = np.zeros(N, dtype=old.dtype)
    if rank != 0:
        req1 = comm.Irecv(upper_row, source=rank - 1, tag=0)
        req2 = comm.Isend(old[:N], dest=rank - 1, tag=1)
        reqs.extend([req1, req2])
        
    if rank != size - 1:
        req1 = comm.Irecv(lower_row, source=rank + 1, tag=1)
        req2 = comm.Isend(old[N*(local_height-1):], dest=rank + 1, tag=0)
        reqs.extend([req1, req2])

    MPI.Request.Waitall(reqs)

    dx2 = dx * dx
    dy2 = dy * dy
    
    for y in range(local_height):
        for x in range(N):
            index = get_index(x, y, N)
            center = old[index]
            val_up    = get_value(old, upper_row, lower_row, x, y-1, N)
            val_down  = get_value(old, upper_row, lower_row, x, y+1, N)
            val_right = get_value(old, upper_row, lower_row, x+1, y, N)
            val_left  = get_value(old, upper_row, lower_row, x-1, y, N)            
            
            d2_dx2 = (val_right - 2*center + val_left) / dx2
            d2_dy2 = (val_up    - 2*center + val_down) / dy2
            local_data[index] = center + dt * (d2_dx2 + d2_dy2)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=100, help='Problem size (NxN grid)')
parser.add_argument('--steps', type=int, default=1000, help='Number of time steps')
args = parser.parse_args()
N = args.N
STEPS = args.steps

L = 1.0 
dx = L / (N - 1)
dy = L / (N - 1)
dt = 0.2 * (dx * dy)**2 / (dx**2 + dy**2)

start_time = MPI.Wtime()

chunk_size = N // size
start_row = rank * chunk_size
end_row = start_row + chunk_size

if rank == size - 1:
    end_row = N

local_size = N * (end_row - start_row)
local_data = np.zeros(local_size, dtype='d')

matrix = None
counts = None
displacements = None
if rank == 0:
    # Initial condition: a hot spot in the center
    matrix = np.zeros(N*N, dtype='d')
    mid = N // 2
    r = N // 10
    for y in range(mid - r, mid + r):
        for x in range(mid - r, mid + r):
            if 0 <= x < N and 0 <= y < N:
                matrix[y*N + x] = 10.0

    counts = [chunk_size * N] * size
    counts[-1] += (N % size) * N
    displacements = [i * chunk_size * N for i in range(size)]

comm.Scatterv([matrix, counts, displacements, MPI.DOUBLE], local_data, root=0)

for _ in range(STEPS):
    step(local_data, start_row, end_row, N, dx, dy, dt)

comm.Gatherv(local_data, [matrix, counts, displacements, MPI.DOUBLE], root=0)

end_time = MPI.Wtime()
if rank == 0:
    elapsed = end_time - start_time
    print(f"Elapsed time = {elapsed:.6f} seconds")
    print(f"Grid: {N}x{N}, Procs: {size}, Steps: {STEPS}")
    
    csv_file = 'performance.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['problem_size', 'size', 'time_elapsed'])
        writer.writerow([N, size, elapsed])

    if N <= 20:
        print("\nResult Map:")
        for y in range(N):
            for x in range(N):
                val = matrix[y*N + x]
                char = "." if val < 0.1 else ("o" if val < 5.0 else "X")
                print(f"{char} ", end="")
            print()