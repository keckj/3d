#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices=3) out;

uniform int totalLayers;

in VS_GS_VERTEX {
	vec3 vertex_position;
	int instanceID;
} vertex_in[];

out GS_FS_VERTEX {
	float z;
} vertex_out;

void main(void) {
	
	int i;
	for(i=0; i<gl_in.length(); ++i) {
		gl_Layer = vertex_in[0].instanceID;	
		gl_Position = vec4(vertex_in[i].vertex_position,1);
		vertex_out.z = float(vertex_in[0].instanceID)/(totalLayers-1);
		EmitVertex();	
	}
	EndPrimitive();
}
