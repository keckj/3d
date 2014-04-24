#version 330

in VS_FS_VERTEX {
	vec3 pos;
} vertex_in;

layout(std140) uniform generalData {
	vec3 textureSize;
	vec3 voxelGridSize;
	vec3 voxelDim;
};

out vec4 out_colour;

void main (void)
{	
	float y = vertex_in.pos.y/(textureSize.y*voxelDim.y);
	out_colour = clamp(vec4(y,1-y,y+y*y,1),0,1); 
}

