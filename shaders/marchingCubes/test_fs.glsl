
#version 130 

in vec3 pos;
out vec4 out_colour;

uniform sampler3D density;

void main (void)
{	
	out_colour = texture(density , pos); 
}

