import numpy as np

import argparse
import time

from nbody import init_mojo_physics_gpu, run_mojo_physics_gpu
from nbody import init_mojo_physics_cpu, run_mojo_physics_cpu
from nbody import init_mojo_physics_cpu, run_mojo_physics_cpu
from nbody import init_mojo_physics_opt_gpu, run_mojo_physics_opt_gpu
from nbody import cpu_physics


from config import N

class NBodySimulation():
    W = 800
    H = 600

    def __init__(self, physics_func=None, init_func=None):


        self.pos = np.random.rand(N, 2).astype(
            np.float32) * np.array([self.W, self.H], dtype=np.float32)
        self.vel = np.zeros((N, 2), dtype=np.float32)
        self.mass = (np.random.rand(N).astype(np.float32) + 1) * 2

        self.physics_func = physics_func
        if init_func is not None:
            self.model = init_func(self.pos, self.mass)
        else:
            self.model = None

    def bench(self):
        repeat=1000
        times = []

        self.physics_func(self.pos, self.mass, self.model)

        for _ in range(repeat):
            start = time.perf_counter()
            if self.model is not None:
                forces = self.physics_func(self.pos, self.mass, self.model)
            else:
                forces = self.physics_func(self.pos, self.mass)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / repeat
        return avg_time




def handle_args():
    custom_args = ['--cpu', '--mojo_gpu', '--mojo_cpu']
    c_args = []

    for c_arg in custom_args:
            c_args.append(c_arg)


    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--cpu",
        action="store_true",
        help="Run computation on pure Python CPU"
    )
    group.add_argument(
        "--mojo_gpu",
        action="store_true",
        help="Run on Mojo Kernel GPU"
    )
    group.add_argument(
        "--mojo_opt_gpu",
        action="store_true",
        help="Run on Mojo Kernel GPU"
    )
    group.add_argument(
        "--mojo_cpu",
        action="store_true",
        help="Run on Mojo Kernel CPU"
    )

    args = parser.parse_args()

    return args


class CPUSim(NBodySimulation):
    def __init__(self):
        super().__init__(init_func=None, physics_func=cpu_physics)


class MojoGPUSim(NBodySimulation):
    def __init__(self):
        super().__init__(init_func=init_mojo_physics_gpu,
                         physics_func=run_mojo_physics_gpu)

class MojoGPUSim(NBodySimulation):
    def __init__(self):
        super().__init__(init_func=init_mojo_physics_gpu,
                         physics_func=run_mojo_physics_gpu)

class MojoCPUSim(NBodySimulation):
    def __init__(self):
        super().__init__(init_func=init_mojo_physics_cpu,
                         physics_func=run_mojo_physics_cpu)

class MojoGPUOptSim(NBodySimulation):
    def __init__(self):
        super().__init__(init_func=init_mojo_physics_opt_gpu,
                         physics_func=run_mojo_physics_opt_gpu)

def main():
    nbody_args = handle_args()
    if nbody_args.cpu:
        sim = CPUSim()
        t = sim.bench()
    elif nbody_args.mojo_gpu:
        sim = MojoGPUSim()
        t = sim.bench()
    elif nbody_args.mojo_opt_gpu:
        sim = MojoGPUOptSim()
        t = sim.bench()
    elif nbody_args.mojo_cpu:
        sim = MojoCPUSim()
        t = sim.bench()

    flops = 16 * (N**2)
    gflops = flops / t / 1e9

    print(f"N={N}, time={t:.6f}s, GFLOP/s={gflops:.2f}")



if __name__ == "__main__":
    main()
