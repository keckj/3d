#version 130 

in vec2 text2D;
in float height;

in vec3 fogColor;
in float fogFactor;

out vec4 out_colour;

uniform sampler2D texture_1, texture_2, texture_3, texture_4, texture_5;

vec4 interp(float thres1, float thres2, sampler2D text1, sampler2D text2);

vec4 applyFog(in vec4 fragColor);


void main (void)
{	
	if (height == 0)
		out_colour = vec4(0.0f,0.0f,0.0f,1.0f);
	else if (height < 0.1)
		out_colour = texture2D(texture_1, text2D);
	else if(height < 0.3)
		out_colour = interp(0.1, 0.3, texture_1, texture_2);
	else if(height < 0.4)
		out_colour = interp(0.3, 0.4, texture_2, texture_3);
	else if(height < 0.6)
		out_colour = interp(0.4, 0.6, texture_3, texture_4);
	else if(height < 0.75)
		out_colour = texture2D(texture_4, text2D);
	else
		out_colour = interp(0.75, 1.0, texture_4, texture_5);

    out_colour = applyFog(out_colour);
}

vec4 interp(float thres1, float thres2, sampler2D text1, sampler2D text2) {
	float alpha = (height - thres1)/(thres2 - thres1);
	vec4 color = (1-alpha)*texture2D(text1, text2D)
		+ alpha*texture2D(text2, text2D);

	return color;
}

vec4 applyFog(in vec4 fragColor) {
    return mix( vec4(fogColor,1.0), fragColor, fogFactor );
}


