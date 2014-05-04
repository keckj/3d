#version 330

in VS_FS_VERTEX 
{
	float intensity;
} vertex_in;

out vec4 out_colour;

void main (void)
{	
	float i = vertex_in.intensity;
	if(i >= 0)
		out_colour = vec4(i,0.0f,1.0f-i,1.0f);
	else
		out_colour = vec4(0.0f,-i,1.0f+i,1.0f);
}
