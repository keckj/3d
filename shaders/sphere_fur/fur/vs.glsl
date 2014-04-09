#version 150 

in vec3 vertex_position;
in vec2 texCoord;

out VS_GS_VERTEX 
{
	vec3 position;
	vec2 texCoord;
} vertex_out;

void main(void)
{
	vertex_out.position = vertex_position;
	vertex_out.texCoord = texCoord;
}
