
#version 420 core

layout(vertices = 3) out;

uniform float split = 5.0;

void main() {

    gl_TessLevelOuter[0] = split; 
    gl_TessLevelOuter[1] = split;
    gl_TessLevelOuter[2] = split;
    gl_TessLevelInner[0] = split;

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
