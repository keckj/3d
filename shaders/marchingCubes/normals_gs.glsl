#version 330


in VS_GS_VERTEX {
	vec3 position;		
	int instanceID; //pour le passer en gl_layer
} vertex_in[];

out GS_FS_VERTEX {
	flat int instanceID;
} vertex_out;

//passthrough avec maj de gl_Layer
layout (triangles) in;
layout (triangle_strip, max_vertices=3) out;

void main(void) {
	
	for(int i=0; i<3; ++i) {
		vertex_out.instanceID = vertex_in[0].instanceID;
		gl_Layer = vertex_in[0].instanceID;
		gl_Position = vec4(vertex_in[i].position,1);
		EmitVertex();	
	}

	EndPrimitive();
}
