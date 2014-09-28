#version 330

in vec3 pos;
in float intensity;
in int alive;

out VS_FS_VERTEX 
{
	float intensity;
	flat int alive;
} vertex_out;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

void main(void) {
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(pos, 1.0);
	vertex_out.intensity = intensity;
	vertex_out.alive = alive;
}
