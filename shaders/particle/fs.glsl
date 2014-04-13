#version 330 core

in float r2;
out vec4 out_colour;

void main (void)
{	
	float r = (r2-0.2)/0.8;
	out_colour = vec4(0,r,1-r,1);
}

