#version 330 core

in GS_FS_VERTEX {
	float z;
} vertex_in;

out float density;

uniform vec2 textureSize;

float metaball(vec3 xyz, vec3 center) {
	vec3 v = xyz - center;
	return 1/dot(v,v);
}

float snoise(vec2 v);

void main (void)
{	
	float z = vertex_in.z;
	vec3 coord = vec3(gl_FragCoord.xy/textureSize, z);
	coord -= vec3(0.5,0.5,0.5);
	
	vec3 center0 = vec3(-0.25,0,0);
	vec3 center1 = vec3(+0.25,0,0);
	
	vec3 sphere1 = coord - center0;
	vec3 sphere2 = coord - center1;

	float r = 0.2;

	/*density += max(0, r*r - dot(sphere1, sphere1));*/
	/*density += max(0, r*r - dot(sphere2, sphere2));*/

	density -= 30;
	density += metaball(coord, center0);
	density += metaball(coord, center1);
}


