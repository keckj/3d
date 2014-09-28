#version 330

in GS_FS_VERTEX {
	flat float r;
} vertex_in;

out vec4 out_colour;

uniform float rmin = 0.0f;
uniform float rmax = 1.0f;

void main (void)
{	
	float rr = (vertex_in.r-rmin)/rmax;
	vec4 color1 = vec4(0,rr,1-rr,0.1);
	vec4 color2 = vec4(0,rr,1-rr,0.8);

	vec2 coords = gl_PointCoord.st - vec2(0.5);

	float d = dot(coords, coords);

	if(d > 0.25)
		discard;
	
	out_colour = mix(color1, color2, smoothstep(0.1,0.25,d));
}
