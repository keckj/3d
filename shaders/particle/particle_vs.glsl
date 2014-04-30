#version 330

in float x;
in float y;
in float z;
in float r;
in int kill;

flat out float r2;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

void main(void) {
	
	vec4 pos = projectionMatrix * viewMatrix * modelMatrix * vec4(x,y,z,1);
	gl_PointSize = (1.0-pos.z/pos.w)* r * 1000.0;
	gl_Position = pos;

	r2=r;
}
