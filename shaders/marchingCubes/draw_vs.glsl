#version 330

in vec3 vertex_position;

out VS_FS_VERTEX {
	vec3 pos;
} vertex_out;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main (void)
{	
	vertex_out.pos = vertex_position;
	gl_Position = projectionMatrix * viewMatrix * vec4(vertex_position,1);
}

