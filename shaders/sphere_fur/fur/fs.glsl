#version 330 core

out vec4 out_colour;

uniform sampler2D noiseTexture;

in GS_FS_VERTEX 
{
	vec3 normal;
	vec3 colour;
	vec2 texCoord;
	flat float fur_strength;
} fragment_in;

void main (void)
{	
	
	float I;

	if(fragment_in.fur_strength == 0.0) {
		I = 1.0;
	}
	else {
		I = fragment_in.fur_strength*texture2D(noiseTexture, fragment_in.texCoord).x;
	}
		
	out_colour = vec4(fragment_in.colour,1)*vec4(1,1,1,I);
}

