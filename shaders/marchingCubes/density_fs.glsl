#version 330 core

in GS_FS_VERTEX {
	float z;
} vertex_in;

out float density;

uniform vec2 textureSize;

void main (void)
{	
	float z = vertex_in.z;
	vec3 coord = vec3(gl_FragCoord.xy/textureSize, z);
	coord -= vec3(0.5,0.5,0.5);
	
	float r = 0.4;
	density = r*r - dot(coord, coord);
}

