import moderngl
import moderngl_window as mglw
import numpy as np
from config import N, dt


class NBodySimulation(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "GPU Rendered N-body Simulation"
    window_size = (800, 600)
    aspect_ratio = 800 / 600
    resizable = True

    resource_dir = "shaders"   # tell moderngl_window where to load shaders from

    def __init__(self, physics_func=None, init_func=None,  **kwargs):
        super().__init__(**kwargs)

        self.wnd.swap_interval = 0

        self.pos = np.random.rand(N, 2).astype(
            np.float32) * np.array([self.wnd.width, self.wnd.height], dtype=np.float32)
        self.vel = np.zeros((N, 2), dtype=np.float32)
        self.mass = (np.random.rand(N).astype(np.float32) + 1) * 2

        self.pos[0,0] = 100
        self.pos[1,0] = 200
        self.pos[0,1] = 100
        self.pos[1,1] = 200

        self.frame_counter = 0

        self.physics_func = physics_func
        if init_func is not None:
            self.model = init_func(self.pos, self.mass)
        else:
            self.model = None

        # Vertex buffer object N * 2 * 4(bytes)
        self.vbo = self.ctx.buffer(reserve=int(N * 2 * 4), dynamic=True)

        self.vbo_array = np.zeros((N, 2), dtype=np.float32)

        # Shader program
        self.prog = self.load_program(
            vertex_shader="nbody.vert",
            fragment_shader="nbody.frag"
        )

        self.prog["screen_size"].value = (self.wnd.width, self.wnd.height)

        vao_content = [(self.vbo, "2f", "in_pos")]
        self.vao = self.ctx.vertex_array(self.prog, vao_content)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def physics_step(self):
        if self.model is not None:
            forces = self.physics_func(self.pos, self.mass, self.model)
        else:
            forces = self.physics_func(self.pos, self.mass)

        # Vectorized velocity and position update
        # mass[:, None] broadcasts over 2D
        self.vel += dt * forces / self.mass[:, None]
        self.pos += dt * self.vel

        # Wrap around screen
        self.pos[:, 0] %= self.wnd.width
        self.pos[:, 1] %= self.wnd.height

    def on_render(self, time: float, frame_time: float):
        self.frame_counter += 1
        # Update physics
        self.physics_step()
        if self.frame_counter % 2 == 0:
            # Upload positions to GPU
            np.copyto(self.vbo_array, self.pos)
            self.vbo.write(self.vbo_array)

        # Draw
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.POINTS)
