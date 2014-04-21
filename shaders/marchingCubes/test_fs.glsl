
#version 130 

in vec3 pos;
out vec4 out_colour;

uniform sampler3D density;

void main (void)
{	
	float val = texture(density , pos); 

	if(val < 0.0)
		out_colour = vec4(0,0,1,1);
	else
		out_colour = vec4(1,0,0,1);
}

