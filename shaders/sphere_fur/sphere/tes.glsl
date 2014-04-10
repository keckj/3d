
#version 420 core

layout(triangles, equal_spacing, ccw) in;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

out TES_GS_VERTEX 
{
	vec3 position;
	vec2 texCoord;
} vertex_out;

void main() {
		
	float alpha = 1.0;
	vec2 coords[3];
	coords[0] = vec2(0.0*alpha, 0.0*alpha);
	coords[1] = vec2(1.0*alpha, 0.0*alpha);
	coords[2] = vec2(1.0*alpha, 1.0*alpha);

	vec3 normal = normalize(
		(gl_TessCoord.x * gl_in[0].gl_Position
		+ gl_TessCoord.y * gl_in[1].gl_Position
		+ gl_TessCoord.z * gl_in[2].gl_Position
		).xyz);

	vertex_out.position = normal;
	vertex_out.texCoord = coords[0]*gl_TessCoord.x + coords[1]*gl_TessCoord.y + coords[2]*gl_TessCoord.z;
}

