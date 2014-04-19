
#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices=90) out;

in TES_GS_VERTEX
{
	vec2 texCoord;
	mat4 transformationMatrix;
} vertex_in[];


out GS_FS_VERTEX
{
	vec3 normal;
	vec3 colour;
	vec2 texCoord;
	flat float fur_strength;
} vertex_out;

void main() {
	int i, j;
	int layers = 30;
	float fur_depth = 0.2;
	float dr = 1.0/layers;
	float d = 0.0;
	
	for(j=0; j<layers; j++) {
		for(i=0; i<3; i++) { //for each points
			vec3 n = gl_in[i].gl_Position.xyz;
			vertex_out.normal = n;
			vertex_out.colour = vec3(0, 1-d, d);
			vertex_out.fur_strength = 1.0 - d;
			vertex_out.texCoord = vertex_in[i].texCoord;
			gl_Position = vertex_in[i].transformationMatrix*(gl_in[i].gl_Position + vec4(n*d*fur_depth,0.0));
			EmitVertex();
		}
		d+=dr;
		EndPrimitive();
	}
}
