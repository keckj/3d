#version 130 

in vec3 viewDir;

out vec4 out_colour;

uniform samplerCube cubemap;

void main (void)
{	
	out_colour = texture(cubemap, viewDir); 
}

