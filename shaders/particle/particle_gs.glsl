
#version 330

in VS_GS_VERTEX {
	vec4 pos;
	uint kill;
	float r;
} vertex_in[];

out GS_FS_VERTEX {
	flat float r;
} vertex_out;

layout(points) in;
layout(points, max_vertices=1) out; 

void main(void) {
	
	vec4 pos = vertex_in[0].pos;
	float r = vertex_in[0].r;
	if(vertex_in[0].kill == 0u) {
		gl_Position = pos;
		gl_PointSize = (1.0-pos.z/pos.w)* r * 1000.0;
		vertex_out.r = r;
		EmitVertex();
	}
}
