#version 330 core 

in float x;
in float y;
in float z;
in float r;
in int kill;

out float r2;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

void main(void) {
	
	
	vec3 vertex_position = vec3(x,y,z);
	mat4 transformationMatrix = projectionMatrix * viewMatrix * modelMatrix;

	gl_PointSize = 20.0*r;
	gl_Position = transformationMatrix * vec4(vertex_position, 1.0f);
	r2=r;
}

