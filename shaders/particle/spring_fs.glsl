#version 330

in VS_FS_VERTEX 
{
	float intensity;
	flat int alive;
} vertex_in;

out vec4 out_colour;

void main (void)
{	
	out_colour = vec4(vertex_in.intensity,0.0f,1.0f-vertex_in.intensity,1.0f);
}
