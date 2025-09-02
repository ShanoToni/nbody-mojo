from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import sqrt
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async

# ANCHOR: physics_kernel
alias TPB = 32

fn physics_kernel[
    pos_layout: Layout,
    mass_layout: Layout,
    out_layout: Layout,
    input_size: Int,
    grav_const: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    pos: LayoutTensor[mut=True, dtype, pos_layout],
    mass: LayoutTensor[mut=True, dtype, mass_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # ADD ME
    if global_i < input_size:
        for i in range(input_size):
            if i == global_i:
                continue

            dx = pos[global_i, 0] - pos[i, 0]
            dy = pos[global_i, 1] - pos[i, 1]

            dist2 = dx * dx + dy * dy + 1e-5
            dist = sqrt(dist2)

            force = grav_const * mass[global_i] * mass[i] / dist2

            fx = -force * dx / dist
            fy = -force * dy / dist

            output[global_i, 0] += fx
            output[global_i, 1] += fy


fn physics_kernel_optimized[
    pos_layout: Layout,      # row_major(N,2)
    mass_layout: Layout,     # row_major(N)
    out_layout: Layout,      # row_major(N,2)
    input_size: Int,         # N
    grav_const: Float32,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    pos:    LayoutTensor[mut=True, dtype, pos_layout],
    mass:   LayoutTensor[mut=True, dtype, mass_layout],
):

    local_i  = thread_idx.x
    global_i = block_idx.x * TPB + thread_idx.x
    if global_i >= input_size:
        return

    pos_sh  = tb[dtype]().row_major[TPB, 2]().shared().alloc()
    mass_sh = tb[dtype]().row_major[TPB]().shared().alloc()

    alias thread_layout_pos  = Layout.row_major(TPB, 2)
    alias thread_layout_mass = Layout.row_major(TPB)

    curr_x    = pos[global_i, 0]
    curr_y    = pos[global_i, 1]
    curr_mass = mass[global_i]

    var fx: output.element_type = 0.0
    var fy: output.element_type = 0.0

    num_tiles: Int = rebind[Int]((input_size + TPB - 1) // TPB)

    G: output.element_type = rebind[Scalar[dtype]](grav_const)

    for tile in range(num_tiles):
        base: Int = tile * TPB

        pos_tile  = pos.tile[TPB, 2](tile, 0)
        mass_tile = mass.tile[TPB](tile)

        copy_dram_to_sram_async[
            thread_layout=thread_layout_pos,
            num_threads=TPB,
            block_dim_count=1,
        ](pos_sh, pos_tile)

        copy_dram_to_sram_async[
            thread_layout=thread_layout_mass,
            num_threads=TPB,
            block_dim_count=1,
        ](mass_sh, mass_tile)

        async_copy_wait_all()
        barrier()

        tile_cols: Int = min(TPB, input_size - base)

        for j in range(tile_cols):
            other: Int = base + j
            if other == global_i:
                continue

            px = pos_sh[j, 0]
            py = pos_sh[j, 1]
            m  = mass_sh[j]

            dx = curr_x - px
            dy = curr_y - py

            dist2 = dx * dx + dy * dy + 1e-5
            dist  = sqrt(dist2)

            force = G * curr_mass * m / dist2
            fx += -force * dx / dist
            fy += -force * dy / dist

        barrier()

    # Write result into the output tile corresponding to this block
    out_tile = output.tile[TPB, 2](block_idx.x, 0)
    out_tile[local_i, 0] = fx
    out_tile[local_i, 1] = fy




from sys import simd_width_of, argv, align_of
from utils import IndexList
from algorithm.functional import elementwise, vectorize

alias SIMD_WIDTH = 1
#simdwidthof[DType.float32]()

# ANCHOR_END: physics_kernel
fn physics_cpu[
    pos_layout: Layout,
    mass_layout: Layout,
    out_layout: Layout,
    input_size: Int,
    grav_const: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, out_layout, MutableAnyOrigin],
    pos: LayoutTensor[mut=True, dtype, pos_layout, MutableAnyOrigin],
    mass: LayoutTensor[mut=True, dtype, mass_layout, MutableAnyOrigin],
) raises:

    @parameter
    @always_inline
    fn compute_physics[width: Int, rank: Int, alignment: Int = align_of[dtype]()](indices: IndexList[rank]) capturing -> None:
        i = indices[0]
        j = indices[1]

        if i != j:
            # Load positions
            pos_x_i = pos.load[width](i, 0)
            pos_y_i = pos.load[width](i, 1)

            pos_x_j = pos.load[width](j, 0)
            pos_y_j = pos.load[width](j, 1)

            # Load masses
            mass_i = mass.load[width](i, 0)
            mass_j = mass.load[width](j, 0)

            # Compute distances
            dx = pos_x_i - pos_x_j
            dy = pos_y_i - pos_y_j
            dist2 = dx * dx + dy * dy + 1e-5
            dist = dist2**0.5

            # Compute force
            force_mag = mass_i * mass_j / dist2
            fx = -force_mag * dx / dist
            fy = -force_mag * dy / dist

            forces_x_old: SIMD[dtype, width] = output.load[width](i, 0)
            forces_y_old: SIMD[dtype, width] = output.load[width](i, 1)

            output.store[width](i, 0, fx + forces_x_old)
            output.store[width](i, 1, fy + forces_y_old)


    elementwise[compute_physics, 1, target="cpu"](IndexList[2](input_size, input_size))



# ANCHOR: physics_custom_op
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer


@compiler.register("physics")
struct PhysicsCustomOp:
    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
        N: Int,
        Grav_Const: Int,
        Optimized: Bool,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank = 2],
        poss: InputTensor[rank = 2],
        mass: InputTensor[rank = 1],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:

        alias pos_layout = Layout.row_major(N, 2)
        alias output_layout = Layout.row_major(N, 2)
        alias mass_layout = Layout.row_major(N)

        output_tensor = rebind[
            LayoutTensor[mut=True, dtype, output_layout, MutableAnyOrigin]
        ](output.to_layout_tensor())

        pos_tensor = rebind[
            LayoutTensor[mut=True, dtype, pos_layout, MutableAnyOrigin]
        ](poss.to_layout_tensor())

        mass_tensor = rebind[
            LayoutTensor[mut=True, dtype, mass_layout, MutableAnyOrigin]
        ](mass.to_layout_tensor())

        @parameter
        if target == "gpu" and Optimized == False:
            gpu_ctx = ctx.get_device_context()
            gpu_ctx.enqueue_function[
                physics_kernel[
                    pos_layout, mass_layout, output_layout, N, Grav_Const
                ]
            ](
                output_tensor,
                pos_tensor,
                mass_tensor,
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=(TPB, 1),
            )

        elif target == "gpu" and Optimized == True:
            gpu_ctx = ctx.get_device_context()
            gpu_ctx.enqueue_function[
                physics_kernel_optimized[
                    pos_layout, mass_layout, output_layout, N, Grav_Const
                ]
            ](
                output_tensor,
                pos_tensor,
                mass_tensor,
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=(TPB, 1),
            )

        elif target == "cpu":
            # we can fallback to CPU

            for i in range(N):
                output_tensor[i,0] = 0.0
                output_tensor[i,1] = 0.0

            cpu_ctx = ctx.get_device_context()
            physics_cpu[
                    pos_layout, mass_layout, output_layout, input_size=N, grav_const=Grav_Const, dtype=dtype
                ](
                    output_tensor, pos_tensor, mass_tensor
                )
        else:
            raise Error("Unsupported target: " + target)


# ANCHOR_END: physics_custom_op
