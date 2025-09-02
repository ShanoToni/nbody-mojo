#version 330

in vec2 in_pos;
uniform vec2 screen_size;

void main() {
    vec2 pos = in_pos / screen_size * 2.0 - 1.0;
    gl_Position = vec4(pos.x, -pos.y, 0.0, 1.0);
    gl_PointSize = 4.0;
}