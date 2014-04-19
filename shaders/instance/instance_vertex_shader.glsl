#version 420 core 

in vec3 vertex_position;
in float offsetX;
in float offsetY;
in float offsetZ;
in float radius;

out mat4 transformationMatrix;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

void main(void)
{
	mat4 modelMatrix = mat4(radius,0.0f,0.0f,0.0f,
			   0.0f,radius,0.0f,0.0f,
			   0.0f,0.0f,radius,0.0f,
			   offsetX, offsetY, offsetZ, 1.0f);
	
	transformationMatrix = projectionMatrix * viewMatrix * modelMatrix;

	gl_Position = vec4(vertex_position, 1.0f);
}

