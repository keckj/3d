#version 130 

in vec3 vertex_position;
in vec3 vertex_colour;
in vec3 vertex_normal;

uniform mat4 modelMatrix = mat4(1,0,0,0,
								0,1,0,0,
								0,0,1,0,
								0,0,0,1);
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

out vec3 color;

void main(void)
{
	mat4 transformationMatrix = projectionMatrix*viewMatrix*modelMatrix;
        color = vertex_colour;
	gl_Position = transformationMatrix * vec4(vertex_position, 1.0);
}

