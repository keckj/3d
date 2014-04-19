#version 130 

in vec3 colour;
out vec4 out_colour;

uniform vec3 color = vec3(1.0,0.0,0.0);

void main (void)
{	
	out_colour = vec4(color,0.0); 
}

