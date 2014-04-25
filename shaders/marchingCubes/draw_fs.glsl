#version 330

in VS_FS_VERTEX {
	vec3 pos;
} vertex_in;

layout(std140) uniform generalData {
	vec3 textureSize;
	vec3 voxelGridSize;
	vec3 voxelDim;
};

uniform sampler3D normals_occlusion;

vec3 sourcePos = vec3(0.5,1,1);

out vec4 out_colour;

void main (void)
{	
	vec3 texCoord = vertex_in.pos/(textureSize*voxelDim);
	vec4 normal_occlusion = texture(normals_occlusion, texCoord);
	vec3 normal = normal_occlusion.xyz; 
	float occlusion = normal_occlusion.w;

	float y = vertex_in.pos.y/(textureSize.y*voxelDim.y);
	
	if(texCoord.x > 0.5)
		out_colour = clamp(vec4(y,1-y,y+y*y,1)*(0.5+clamp(occlusion,0,1)*dot(normal, normalize(sourcePos - texCoord))),0,1);
	else
		out_colour = clamp(vec4(y,1-y,y+y*y,1)*(0.2+dot(normal, normalize(sourcePos - texCoord))),0,1);
}

