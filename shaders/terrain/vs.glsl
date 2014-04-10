#version 130 

in vec3 vertex_position;
in vec3 vertex_colour;

out vec2 text2D;
out float height;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

void main(void)
{
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertex_position, 1.0);
	text2D = vertex_colour.xy;
	height = vertex_position.z/255.0f;
}

