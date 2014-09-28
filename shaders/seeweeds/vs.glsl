#version 330

in vec3 pos;
in float intensity;

out VS_FS_VERTEX 
{
        vec3 pos;
	float intensity;
} vertex_out;

layout(std140) uniform projectionView {
	mat4 projectionMatrix;
	mat4 viewMatrix;
	vec3 cameraPos;
	vec3 cameraDir;
	vec3 cameraUp;
	vec3 cameraRight;
};

uniform mat4 modelMatrix;

void main(void) {
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(pos, 1.0);
        vertex_out.pos = pos;
	vertex_out.intensity = intensity;
}
