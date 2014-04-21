#version 330 core

in GS_FS_VERTEX {
	float z;
} vertex_in;

/*out vec4 out_colour;*/
out float density;

uniform vec2 textureSize;

void main (void)
{	
	float z = vertex_in.z;
	vec3 coord = vec3(gl_FragCoord.xy/textureSize, z);
	coord -= vec3(0.5,0.5,0.5);


	//if(coord.x < 0.0) {
	//	if(coord.y<0.0)
	//		out_colour = vec4(1.0-z, z, z, 1.0);
	//	else
	//		out_colour = vec4(1.0, z, 1-z, 1.0);
	//}
	//else {
	//	if(coord.y<0.0)
	//		out_colour = vec4(z, 1-z, 1.0, 1.0);
	//	else
	//		out_colour = vec4(z, 1, 1-z, 1.0);
	//}
	
	float r = 0.4;
	density = r*r - dot(coord, coord); 
}

