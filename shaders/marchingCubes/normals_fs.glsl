#version 330 core

out vec4 normal_occlusion;

in GS_FS_VERTEX {
	flat int instanceID;
} vertex_in;


layout(std140) uniform generalData {
	vec3 textureSize;
	vec3 voxelGridSize;
	vec3 voxelDim;
};

layout(std140) uniform poissonDistributions {
	vec3 poisson256[256];
	vec3 poisson128[128];
	vec3 poisson64[64];
	vec3 poisson32[32];
};

uniform sampler3D density;


vec3 computeGradient(vec3 pos, vec4 step);
float computeAmbiantOcclusion(vec3 pos);

void main(void)
{
	vec3 pos = vec3(gl_FragCoord.xy/textureSize.xy, (vertex_in.instanceID+0.5)/textureSize.z);
	vec4 step = vec4(1/voxelGridSize,0);

	vec3 normal = normalize(- computeGradient(pos, step));
	float occlusion = computeAmbiantOcclusion(pos); 

	normal_occlusion = vec4(normal, occlusion);
}


//calcul du gradient de la densité au point pos
vec3 computeGradient(vec3 pos, vec4 step) {
	
	vec3 gradient = vec3(
					(texture(density, pos + step.xww) - texture(density, pos - step.xww)).x,
				    (texture(density, pos + step.wyw) - texture(density, pos - step.wyw)).x,
				    (texture(density, pos + step.wwz) - texture(density, pos - step.wwz)).x
				 );

	return gradient;
}


//calcul de l'occlusion ambiante en échantillonant la
//densité le long de 32 vecteurs distribués selon une 
//loi de Poisson sur une sphère
float computeAmbiantOcclusion(vec3 pos) {
	
	int ray, step;

	float factor = 0.2; //20% of map
	float shortStep = factor/32;
	float longStep = factor/4;

	float visibility = 0.0f;
	
	for(ray=0; ray<32; ray++) {

		vec3 dir = poisson32[ray]; //normalized
		float ray_visibility = 1.0f;

		//sample courte portée
		for(step = 1; step <= 16; step++) {
			float d = texture(density, pos + step * shortStep * dir);	
			ray_visibility *= clamp(d * 9999,0,1);  
		}
		
		//sample longue portée
		for(step = 1; step <= 4; step++) {
			float d = texture(density, pos + step * longStep * dir);	
			ray_visibility *= clamp(d * 9999,0,1);  
		}

		visibility += ray_visibility;
	}

	//on retourne l'occlusion
	return (1 - visibility/32.0);
}

