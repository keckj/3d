
#version 420 core

layout(vertices = 3) out;

in mat4 transformationMatrix[];
out mat4 out_transformationMatrix[];

void main() {

    gl_TessLevelOuter[0] = 5.0; 
    gl_TessLevelOuter[1] = 5.0;
    gl_TessLevelOuter[2] = 5.0;
    gl_TessLevelInner[0] = 5.0;

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    out_transformationMatrix[gl_InvocationID] = transformationMatrix[gl_InvocationID];
}
