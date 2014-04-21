#version 330 core

in vec3 vertex_position;

out VS_GS_VERTEX {
	vec3 vertex_position;
	int instanceID;
} vertex_out;

void main(void)
{
	vertex_out.vertex_position = vertex_position;
	vertex_out.instanceID = gl_InstanceID;
}

