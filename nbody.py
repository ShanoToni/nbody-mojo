import moderngl_window as mglw
import sys
from simulation import NBodySimulation
from config import N, G, dt

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.dtype import DType
import argparse


def init_mojo_physics(
    pos, mass,
    optimized = False,
    device: Device = Accelerator(),
    session: InferenceSession = InferenceSession(devices=[Accelerator()]),
):

    # Create driver tensors from the input arrays and move them to the target device
    pos = np.asarray(pos, dtype=np.float32).reshape(N, 2)
    mass = np.asarray(mass, dtype=np.float32)
    pos_tensor = Tensor.from_numpy(pos).to(device)
    mass_tensor = Tensor.from_numpy(mass).to(device)

    # Path to the directory containing our Mojo operations
    mojo_kernels = Path(__file__).parent / "op"

    # Configure our graph with the custom physics operation
    with Graph(
        "physics_graph",
        input_types=[
            TensorType(
                DType.float32,
                shape=pos_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,
                shape=mass_tensor.shape,
                device=DeviceRef.from_device(device),
            )
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Define inputs to the graph
        pos_value, mass_value = graph.inputs

        # The output shape is the same as the input
        output = ops.custom(
            name="physics",
            device=DeviceRef.from_device(device),
            values=[pos_value, mass_value],
            out_types=[
                TensorType(
                    dtype=pos_value.tensor.dtype,
                    shape=pos_value.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={
                "N": N,
                "Grav_Const": G,
                "Optimized": optimized,
                "dtype": DType.float32,
            },
        )[0].tensor
        graph.output(output)

    # Compile the graph
    print("Compiling Physics graph...")
    return session.load(graph)

def init_mojo_physics_gpu(
    pos, mass
):
    return init_mojo_physics(pos, mass, False, Accelerator(), InferenceSession(devices=[Accelerator()]))

def init_mojo_physics_opt_gpu(
    pos, mass
):
    return init_mojo_physics(pos, mass, True, Accelerator(), InferenceSession(devices=[Accelerator()]))

def init_mojo_physics_cpu(
    pos, mass
):
    return init_mojo_physics(pos, mass, False, CPU(), InferenceSession(devices=[CPU()]))



def run_mojo_physics(
    pos, mass,
    model=None,
    device: Device = Accelerator(),
) -> Tensor:
    dtype = DType.float32

    # Create driver tensors from the input arrays and move them to the target device
    pos = np.asarray(pos, dtype=np.float32).reshape(N, 2)
    mass = np.asarray(mass, dtype=np.float32)

    pos_tensor = Tensor.from_numpy(pos).to(device)
    mass_tensor = Tensor.from_numpy(mass).to(device)

    # Execute the operation
    result = model.execute(pos_tensor, mass_tensor)[0]

    # Copy values back to the CPU to be read
    if device == Accelerator():
        return result.to(CPU()).to_numpy()
    elif device == CPU():
        return result.to_numpy()



def run_mojo_physics_gpu(
    pos, mass, model=None,
) -> Tensor:
    dtype = DType.float32

    return run_mojo_physics(pos, mass, model, Accelerator())

def run_mojo_physics_opt_gpu(
    pos, mass, model=None,
) -> Tensor:
    dtype = DType.float32

    return run_mojo_physics(pos, mass, model, Accelerator())

def run_mojo_physics_cpu(
    pos, mass, model=None,
) -> Tensor:
    dtype = DType.float32
    return run_mojo_physics(pos, mass, model, CPU())



def cpu_physics(pos, mass, model=None):
    N = pos.shape[0]

    # Compute pairwise differences: pos_j - pos_i
    dx = pos[:, 0][:, np.newaxis] - pos[:, 0]  # shape (N, N)
    dy = pos[:, 1][:, np.newaxis] - pos[:, 1]  # shape (N, N)

    # Distance squared with small epsilon to avoid division by zero
    dist2 = dx**2 + dy**2 + 1e-5
    dist = np.sqrt(dist2)

    # Force magnitude: G * m_i * m_j / dist^2
    force_mag = G * mass[:, np.newaxis] * mass[np.newaxis, :] / dist2

    # Normalize vector and multiply by magnitude
    fx = -force_mag * dx / dist
    fy = -force_mag * dy / dist

    # Zero out self-force (i == j)
    np.fill_diagonal(fx, 0.0)
    np.fill_diagonal(fy, 0.0)

    # Sum forces from all j on each i
    forces = np.stack([fx.sum(axis=1), fy.sum(axis=1)], axis=1)  # shape (N, 2)
    return forces.astype(np.float32)


class CPUSim(NBodySimulation):
    def __init__(self, **kwargs):
        super().__init__(init_func=None, physics_func=cpu_physics, **kwargs)


class MojoGPUSim(NBodySimulation):
    def __init__(self, **kwargs):
        super().__init__(init_func=init_mojo_physics_gpu,
                         physics_func=run_mojo_physics_gpu, **kwargs)

class MojoGPUOptSim(NBodySimulation):
    def __init__(self, **kwargs):
        super().__init__(init_func=init_mojo_physics_opt_gpu,
                         physics_func=run_mojo_physics_opt_gpu, **kwargs)

class MojoCPUSim(NBodySimulation):
    def __init__(self, **kwargs):
        super().__init__(init_func=init_mojo_physics_cpu,
                         physics_func=run_mojo_physics_cpu, **kwargs)

def handle_args():
    mglw_args = sys.argv[1:]
    custom_args = ['--cpu', '--mojo_gpu', '--mojo_cpu', '--mojo_opt_gpu']
    c_args = []

    for c_arg in custom_args:
        if c_arg in mglw_args:
            c_args.append(c_arg)
            mglw_args.remove(c_arg)


    mglw_args.append('-r')
    mglw_args.append('True')

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

    args = mglw.parse_args(parser=parser)

    nbody_args = args
    return mglw_args, nbody_args


def main():
    mglw_args, nbody_args = handle_args()

    if nbody_args.cpu:
        mglw.run_window_config(CPUSim, args=mglw_args)
    elif nbody_args.mojo_gpu:
        mglw.run_window_config(MojoGPUSim, args=mglw_args)
    elif nbody_args.mojo_opt_gpu:
        mglw.run_window_config(MojoGPUOptSim, args=mglw_args)
    elif nbody_args.mojo_cpu:
        mglw.run_window_config(MojoCPUSim, args=mglw_args)


if __name__ == "__main__":
    main()
