
#version 140 

in vec3 vertex_position;

out vec3 pos;

uniform mat4 modelMatrix = mat4(1,0,0,0,
								0,1,0,0,
								0,0,1,0,
								0,0,0,1);
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

uniform int layers;

void main(void)
{
	mat4 transformationMatrix = projectionMatrix*viewMatrix*modelMatrix;
	float delta = 1.0f/(layers-1);
	
	gl_Position = transformationMatrix * vec4(vertex_position.xy, delta*gl_InstanceID, 1.0);
	pos = vec3(vertex_position.xy, (gl_InstanceID + 0.5)/layers );
}

