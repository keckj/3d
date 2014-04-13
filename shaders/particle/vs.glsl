#version 330

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
	
	
	vec4 cameraPos = viewMatrix*modelMatrix*vec4(x,y,z,1);

	float d = -cameraPos.z;
	gl_PointSize = 1000.0*r/d;
	gl_Position = projectionMatrix * cameraPos;

	r2=r;
}
