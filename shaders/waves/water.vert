#version 150

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 invView;

uniform float time;
uniform float deltaX;
uniform float deltaZ;

uniform	float fogDensity = 0.05;
uniform	float underWaterFogEnd = 30.0;
uniform vec3 sunDir = vec3(100.0,40.0,-50.0);

in vec3 position;

out vec3 fPosition;
out vec3 fNormal;

out vec3 fogColor;
out float fogFactor;


vec3 underWaterFogColor = vec3(57.0/256.0,88.0/256.0,121.0/256.0);
vec3 cameraPos;
float waterHeight = modelMatrix[3][1];
float waveHeight = 0.6;


// Source: github.com/ashima/webgl-noise/blob/master/src/noise2D.glsl

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
    return mod289(((x*34.0)+1.0)*x);
}

float snoise2(vec2 v) {
    const vec4 C = vec4(0.211324865405187, // (3.0-sqrt(3.0))/6.0
            0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
            -0.577350269189626, // -1.0 + 2.0 * C.x
            0.024390243902439); // 1.0 / 41.0
    // First corner
    vec2 i = floor(v + dot(v, C.yy) );
    vec2 x0 = v - i + dot(i, C.xx);

    // Other corners
    vec2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
            + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

    // Compute final noise value at P
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}


void computeFogColor(in vec4 position) {

    vec3 rayDir = normalize(cameraPos - position.xyz);
    float distance = length(cameraPos - position.xyz);
    float hc = cameraPos.y - position.y;    
    float hp = position.y - position.y;    

    if (hc >= 0 && hp >= 0) {
        // Brouillard exponentiel au dessus de l'eau (soleil de la skybox pris en compte)
        fogFactor = exp( -distance*fogDensity );
        float sunFactor = max( dot( rayDir, normalize(sunDir) ), 0.0 );
        fogColor = mix( vec3(0.26,0.26,0.26), // fog
                        vec3(1.0,0.9,0.7), // sun
                        pow(sunFactor,8.0) );
        //fogColor = vec3(0.8,0.8,0.8);
    } else if (hc < 0 && hp < 0) {
        // Brouillard linéaire sous l'eau (début immédiatement devant la caméra)
        fogFactor = clamp((underWaterFogEnd - distance) / underWaterFogEnd, 0.0, 1.0); 
        fogColor = underWaterFogColor; 
    } else if (hc >= 0 && hp < 0) {
        // Brouillard linéaire proportionnel à la longueur traversée sous l'eau
        float d = distance * hp / (hp +  hc);
        fogFactor = clamp((underWaterFogEnd - d) / underWaterFogEnd, 0.0, 1.0); 
        fogColor = underWaterFogColor; 
    } else if (hc < 0 && hp >= 0) {
        // Brouillard linéaire proportionnel à la longueur traversée sous l'eau
        float d = distance * hc / (hp +  hc);
        fogFactor = clamp((underWaterFogEnd - d) / underWaterFogEnd, 0.0, 1.0); 
        fogColor = underWaterFogColor; 
    }
}

float applyNoise(vec2 pos) {
    /*return waveHeight * (0.2*snoise2((vec2(pos.x+3.14,pos.y+7.89)+vec2(time+5.52))*2.0)
                        + snoise2(vec2(pos.x+58.2,pos.y+0.35)+vec2(time+7.78)));*/
    return  waveHeight * (
            0.5*snoise2((pos+time)*2.0/10.0)
            + 0.25*snoise2((pos+time)*4.0/10.0)
            + 0.125*snoise2((pos+time)*8.0/10.0)
            + 0.0615*snoise2((pos+time)*16.0/10.0)
            );
}

vec3 calcNormals(in vec3 pos) {
    vec2 xOffsetPosP = pos.xz + vec2(deltaX/10.0, 0.0);
    vec2 zOffsetPosP = pos.xz + vec2(0.0, deltaZ/10.0);
    float xOffsetHeightP = applyNoise(xOffsetPosP);
    float zOffsetHeightP = applyNoise(zOffsetPosP);
    vec3 modelXOffsetP = vec3(xOffsetPosP.x, xOffsetHeightP, xOffsetPosP.y);
    vec3 modelZOffsetP = vec3(zOffsetPosP.x, zOffsetHeightP, zOffsetPosP.y);

    vec2 xOffsetPosN = pos.xz - vec2(deltaX/10.0, 0.0);
    vec2 zOffsetPosN = pos.xz - vec2(0.0, deltaZ/10.0);
    float xOffsetHeightN = applyNoise(xOffsetPosN);
    float zOffsetHeightN = applyNoise(zOffsetPosN);
    vec3 modelXOffsetN = vec3(xOffsetPosN.x, xOffsetHeightN, xOffsetPosN.y);
    vec3 modelZOffsetN = vec3(zOffsetPosN.x, zOffsetHeightN, zOffsetPosN.y);

    vec3 modelXGrad = modelXOffsetP - modelXOffsetN;
    vec3 modelZGrad = modelZOffsetP - modelZOffsetN;
    return -normalize(cross(modelXGrad, modelZGrad));
}


void main()
{
    // Dernière colonne
    //cameraPos = vec3(invView[3]);
    cameraPos = -viewMatrix[3].xyz * mat3(viewMatrix);

    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    worldPos.y += applyNoise(vec2(worldPos.x, worldPos.z));
    fPosition = worldPos.xyz;
    fNormal = calcNormals(fPosition);
    computeFogColor(worldPos);
    vec4 viewPos = viewMatrix * worldPos;
    gl_Position = projectionMatrix * viewPos;
}
