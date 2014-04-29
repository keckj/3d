#version 130

uniform mat4 viewMatrix;

uniform float time;
uniform vec3 sunDir = vec3(100.0,100.0,0.0);
uniform samplerCube cubeMapTexture;

in vec3 fPosition;
in vec3 fNormal;

in vec3 fogColor;
in float fogFactor;

out vec4 out_color;


vec4 applyFog(in vec4 fragColor) {
    return mix( vec4(fogColor,1.0), fragColor, fogFactor );
}

void main()
{
    /* BEGIN LIGHT */  
    vec3 l = normalize(sunDir);
    vec4 specular = vec4(0.05,0.05,0.05,1.0);
    vec4 diffuse = vec4(57.0/256.0,88.0/256.0,121.0/256.0,.5);
    vec4 ambient = diffuse/4.0;
    float shininess = 100.0;

    // set the specular term to black
    vec4 spec = vec4(0.0);

    // normalize both input vectors
    vec3 n = normalize(fNormal);
    vec3 e = normalize(vec3(- viewMatrix * vec4(fPosition,1.0)));

    float lambertTerm  = max(dot(n,l), 0.0);

    // if the vertex is lit compute the specular color
    if (lambertTerm  > 0.0) {
        float intSpec = max(0.0, dot(reflect(-l, n), e));
        spec = specular * pow(intSpec,shininess);
    }
    out_color = max(lambertTerm  *  diffuse + spec, ambient);

    /* END LIGHT */

    /* BEGIN ENVMAP */
    // reflection vector
    vec3 reflectDir = reflect(e, n);

    // lookup into the environment map
    vec3 envColor = vec3(texture(cubeMapTexture, normalize(reflectDir)));

    // mix
    out_color = vec4(mix(envColor, vec3(out_color), 0.5), out_color.a);
    /* END ENVMAP */

    // apply fog
    out_color = applyFog(out_color);
}
