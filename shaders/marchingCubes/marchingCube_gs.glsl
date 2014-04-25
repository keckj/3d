#version 330

layout (points) in; //les points inférieurs gauche des cubes
layout (triangle_strip, max_vertices=15) out; //max 5 triangles en sortie

in VS_GS_VERTEX {
	vec3 uvw; //coordonnées réduites du bord inférieur gauche du cube [0,1]x[0,1]x[0,1]
	vec4 f0123; //les valeurs de la densité aux 8 bords du cube
	vec4 f4567;
	uint caseId; //le cas du MC 0..255
} vertex_in[];

out GS_FS_VERTEX {
	vec3 worldPos;
} vertex_out;

//Nombre de triangle par cas + info sur les 12 cotés
layout (std140) uniform lookupTable {
	uint caseToNumPolys[256]; //nb de triangle par cas
	vec3 edgeStart[12];        //debut des cotés
	vec3 edgeDir[12];          //direction des cotés
	vec4 maskA0123[12];        //masques pour retrouver la densité en fonction du coté
	vec4 maskB0123[12];
	vec4 maskA4567[12];
	vec4 maskB4567[12];
};

//table de triangles
//max 5 triangle par cas
//un triangle = int3(x,y,z) 
//avec x,y,z les cotés du cube touchés par le triangle
layout(std140) uniform triangleTable {
	ivec3 triTable[1280]; //5*256
};

layout(std140) uniform generalData {
	vec3 textureSize;
	vec3 voxelGridSize;
	vec3 voxelDim;
};

vec3 computeTriangleVertex(vec3 uvw, vec4 f0123, vec4 f4567, int edgeNum) {

	//interpolation linéaire entre le point A et B
	//pour trouver le point ou la densité vaut 0
	float dA = dot(f0123, maskA0123[edgeNum]) + dot(f4567, maskA4567[edgeNum]);
	float dB = dot(f0123, maskB0123[edgeNum]) + dot(f4567, maskB4567[edgeNum]);

	float t = clamp(dA/(dA-dB), 0.0, 1.0); //evite les trucs bizarre avec les triangles du bord

	vec3 pos_within_cell = edgeStart[edgeNum] + t*edgeDir[edgeNum]; // [0..1]x[0..1]x[0..1]

	vec3 worldPos = (uvw*voxelGridSize + pos_within_cell)*voxelDim;

	return worldPos;
}

void main(void) {


	int num_polys = int(caseToNumPolys[vertex_in[0].caseId]);
	
	uint tri_table_pos = 5u*vertex_in[0].caseId;

	for(int i=0; i<num_polys; ++i) {

		ivec3 triangleData = triTable[tri_table_pos++];

		vertex_out.worldPos = computeTriangleVertex(
				vertex_in[0].uvw,
				vertex_in[0].f0123, vertex_in[0].f4567,
				triangleData.x
				);
		EmitVertex();	
		
		vertex_out.worldPos = computeTriangleVertex(
				vertex_in[0].uvw,
				vertex_in[0].f0123, vertex_in[0].f4567,
				triangleData.y
				);
		EmitVertex();	
		
		vertex_out.worldPos = computeTriangleVertex(
				vertex_in[0].uvw,
				vertex_in[0].f0123, vertex_in[0].f4567,
				triangleData.z
				);
		EmitVertex();	

		EndPrimitive();
	}
	
}
