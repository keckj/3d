/* Note: couleurs non finales 
 * IMPORTANT : axe vertical = y                 <=========== README
 * fogDensity : uniform
 * underWaterFogEnd : uniform
 * cameraPos : uniform
 * waterHeight : uniform (hauteur moyenne, ça se voit pas avec des petites vagues, ou au pire on évite de filmer à ce niveau)
 * sunDir : uniform (direction de la lumière du soleil, calculée avec depuis la skybox)
 * http://www.iquilezles.org/www/articles/fog/fog.htm
 */
 
/***********************************************************************************/
// Vertex Shader 
/***********************************************************************************/

in vec3 vertexPosition;
uniform	float fogDensity = 0.35;
uniform	float underWaterFogEnd;
uniform vec3 cameraPos;
uniform float waterHeight;
uniform vec3 sunDir;
out vec3 fogColor;
out float fogFactor;

vec3 underWaterFogColor = vec3(0.1,0.2,0.7); // à ajuster

void computeFogColor(in vec4 position) {

    vec3 rayDir = normalize(cameraPos - position.xyz);
    float distance = length(cameraPos - position.xyz);
    float hc = cameraPos.y - waterHeight;
    float hp = position.y - waterHeight;
    
    // On évite les if imbriqués pour des raisons de performances
    if (hc > 0 && hp > 0) {
        // Brouillard exponentiel au dessus de l'eau (soleil de la skybox pris en compte)
        fogFactor = exp( -distance*fogDensity );
        float sunFactor = max( dot( rayDir, normalize(sunDir) ), 0.0 );
        fogColor = mix( vec3(0.5,0.6,0.7), // bluish
                        vec3(1.0,0.9,0.7), // yellowish
                        pow(sunFactor,8.0) );
    } else if (hc < 0 && hp < 0) {
        // Brouillard linéaire sous l'eau (début immédiatement devant la caméra)
        fogFactor = clamp((underWaterFogEnd - distance) / underWaterFogEnd, 0.0, 1.0); 
        fogColor = underWaterFogColor; 
    } else if (hc > 0 && hp < 0) {
        // Brouillard linéaire proportionnel à la longueur traversée sous l'eau
        float d = distance * hp / (hp +  hc);
        fogFactor = clamp((underWaterFogEnd - d) / underWaterFogEnd, 0.0, 1.0); 
        fogColor = underWaterFogColor; 
    } else if (hc < 0 && hp > 0) {
        // Brouillard linéaire proportionnel à la longueur traversée sous l'eau
        float d = distance * hc / (hp +  hc);
        fogFactor = clamp((underWaterFogEnd - d) / underWaterFogEnd, 0.0, 1.0); 
        fogColor = underWaterFogColor; 
    }
}

void main {

    vec4 worldPos = modelMatrix * vertexPosition;
    gl_Position = projMatrix * viewMatrix * worldPos;
 
    computeFogColor(worldPos);
}


/***********************************************************************************/
// Fragment Shader 
/***********************************************************************************/

in vec3 fogColor;
in float fogFactor;


vec4 applyFog(in vec4 fragColor) { // couleur sans brouillard du fragment considéré
    return mix( vec4(fogColor,1.0), fragColor, fogFactor );
}

void main()
{
    vec4 color = [VALEUR CALCULEE];
    vec4 outColor = applyFog(color);
}
