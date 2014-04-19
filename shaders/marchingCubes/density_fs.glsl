#version 330 core

in GS_FS_VERTEX {
	float z;
} vertex_in;

out vec4 out_colour;

uniform vec3 boxSize = vec3(1.0f, 1.0f, 1.0f);

void main (void)
{	
	vec3 worldCoord = vec3(gl_FragCoord.x - 0.5, gl_FragCoord.y - 0.5, vertex_in.z - 0.5) * boxSize;
	float z = vertex_in.z;
	out_colour = vec4(1.0-z,z,0.0,1.0); 
}

