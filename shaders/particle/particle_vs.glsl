#version 330

in float x;
in float y;
in float z;
in float r;
in uint kill;

out VS_GS_VERTEX {
	vec4 pos;
	uint kill;
	float r;
} vertex_out;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

void main(void) {
	
	vec4 pos = projectionMatrix * viewMatrix * modelMatrix * vec4(x,y,z,1);
	vertex_out.pos =  pos;
	vertex_out.kill = kill;
	vertex_out.r = r;
}
