#version 330 core

in vec3 vertex_position;

out VS_GS_VERTEX {
	vec3 position;		
	int instanceID; //pour le passer en gl_layer
} vertex_out;

void main(void) {
	vertex_out.position = vertex_position;	
	vertex_out.instanceID = gl_InstanceID;
}
