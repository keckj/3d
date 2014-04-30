#version 150 

in vec3 vertex_position;

uniform mat4 modelMatrix = mat4(1,0,0,0,
				0,1,0,0,
				0,0,1,0,
				0,0,0,1);

layout(std140) uniform projectionView {
	mat4 projectionMatrix;
	mat4 viewMatrix;
	vec3 cameraPos;
	vec3 cameraDir;
	vec3 cameraUp;
	vec3 cameraRight;
};

out vec3 viewDir;

void main(void)
{
	vec4 worldPos = modelMatrix*vec4(vertex_position,1);

	gl_Position = projectionMatrix * viewMatrix * worldPos;
	viewDir = worldPos.xyz - cameraPos;
}

