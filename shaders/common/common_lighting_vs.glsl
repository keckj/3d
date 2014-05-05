#version 330

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

uniform	float fogDensity = 0.05;
uniform	float underWaterFogEnd = 50.0;
uniform vec3 sunDir = vec3(100.0,40.0,-50.0);

in vec4 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;

out vec4 fPosition;
out vec3 fNormal;
out vec2 fTexCoord;

out vec3 fogColor;
out float fogFactor;

vec3 underWaterFogColor = vec3(57.0/256.0,88.0/256.0,121.0/256.0);
vec3 cameraPos;
float waterHeight = 10.0; // change if necessary


void computeFogColor(in vec4 position) {

    vec3 rayDir = normalize(cameraPos - position.xyz);
    float distance = length(cameraPos - position.xyz);
    float hc = cameraPos.y - waterHeight; // waterHeight approx.    
    float hp = position.y - waterHeight;    

    if (hc >= 0 && hp >= 0) {
        // Brouillard exponentiel au dessus de l'eau (soleil de la skybox pris en compte)
        fogFactor = exp( -distance*fogDensity );
        float sunFactor = max( dot( rayDir, normalize(sunDir) ), 0.0 );
        fogColor = mix( vec3(0.5,0.6,0.7),// vec3(0.8,0.8,0.8), // bluish
                        vec3(1.0,0.9,0.7), // yellowish
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


void main()
{
    cameraPos = -viewMatrix[3].xyz * mat3(viewMatrix);
    //mat3 normalMatrix1 = transpose(inverse(mat3(viewMatrix * modelMatrix)));
    //fNormal = normalize(normalMatrix1 * vertexNormal);
    fNormal = normalize(normalMatrix * vertexNormal);
    fTexCoord = vertexTexCoord;
    fPosition = modelMatrix * vertexPosition;
    computeFogColor(fPosition);
    gl_Position = projectionMatrix * viewMatrix * fPosition;
}
