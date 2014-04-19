
#version 400 core
layout(triangles, invocations=5) in;
layout(triangle_strip, max_vertices=90) out;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

in VS_GS_VERTEX
{
	vec3 position;
	vec2 texCoord;
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
	int invocations = 5;
	int layers = 10;
	int totalLayers = layers*invocations;
	float fur_depth = 0.2;
	float dr = 1.0/totalLayers;
	float d = dr*(layers*gl_InvocationID);

	mat4 transformationMatrix = projectionMatrix*viewMatrix;
	
	for(j=0; j<layers; j++) {
		for(i=0; i<gl_in.length(); i++) { //for each points
			vec3 n = vertex_in[i].position;
			vertex_out.normal = n;
			vertex_out.colour = vec3(0, 0.1+(1-d)*0.9, 0);
			vertex_out.fur_strength = d;
			vertex_out.texCoord = vertex_in[i].texCoord;
			
			gl_Position = transformationMatrix*
			vec4((1+d*fur_depth)*n, 1);
			EmitVertex();
		}
		d+=dr;
		EndPrimitive();
	}
}
