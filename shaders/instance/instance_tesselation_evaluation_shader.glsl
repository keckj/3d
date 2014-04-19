
#version 420 core

layout(triangles, equal_spacing, ccw) in;

in TCS_TEST_VERTEX
{
	in mat4 transformationMatrix;
} vertex_in[];

out TES_GS_VERTEX 
{
	mat4 transformationMatrix;
	vec2 texCoord;
} vertex_out;

void main() {
	
	vec2 coords[3];
	coords[0] = vec2(0.5, 0.0);
	coords[1] = vec2(1.0, 0.0);
	coords[2] = vec2(1.0, 1.0);

	vec3 normal = normalize(
		(gl_TessCoord.x * gl_in[0].gl_Position
		+ gl_TessCoord.y * gl_in[1].gl_Position
		+ gl_TessCoord.z * gl_in[2].gl_Position
		).xyz);

	vertex_out.transformationMatrix = vertex_in[0].transformationMatrix;
	vertex_out.texCoord = coords[0]*gl_TessCoord.x + coords[1]*gl_TessCoord.y + coords[2]*gl_TessCoord.z;
	gl_Position = vec4(normal,1.0);

}

