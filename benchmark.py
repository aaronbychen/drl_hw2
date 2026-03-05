"""
Filename: benchmark.py
Authors: Aaron Chen (517971), Sherlock Zhang (518915)
"""

import time
import torch
import numpy as np
from buffer import PrioritizedReplayBuffer, SumTreePrioritizedReplayBuffer

def run_benchmark():
    device = torch.device('cpu') # prevent gpu parallel distractions
    batch_size = 128
    
    capacities = [10_000, 50_000, 100_000, 500_000]
    
    print(f"{'Capacity':<10} | {'O(N) Old PER (s)':<20} | {'O(logN) SumTree (s)':<20}")
    print("-" * 55)

    for cap in capacities:
        # state_size 4 (CartPole)
        old_buf = PrioritizedReplayBuffer(cap, 0.01, 0.7, 0.4, 4, device)
        new_buf = SumTreePrioritizedReplayBuffer(cap, 0.01, 0.7, 0.4, 4, device)
        
        # fake data
        dummy_transition = (np.zeros(4), 0, 1.0, np.zeros(4), 0)
        for _ in range(cap):
            old_buf.add(dummy_transition)
            new_buf.add(dummy_transition)
            
        # old O(N) Array
        start = time.time()
        for _ in range(1000): # sampling 1000 batch
            old_buf.sample(batch_size)
        time_old = time.time() - start
        
        # new O(logN) SumTree
        start = time.time()
        for _ in range(1000): 
            new_buf.sample(batch_size)
        time_new = time.time() - start
        
        print(f"{cap:<10} | {time_old:<20.4f} | {time_new:<20.4f}")

if __name__ == "__main__":
    run_benchmark()