#version 330 core

uniform mat4 viewMatrix;
uniform mat4 ViewMatrixInv = inverse(viewMatrix);

struct Light {
    bool isEnabled;
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    float constantAttenuation, linearAttenuation, quadraticAttenuation;
    float spotCutoff, spotExponent;
    vec3 spotDirection;
};
const int nLights = 5;
uniform Light lights[nLights];

struct Material {
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
    float transparency;
    bool hasTexture;
};
uniform Material mat;

uniform vec4 scene_ambient = vec4(0.2,0.2,0.2,1.0);

uniform sampler2D diffuseTexture;

in vec4 fPosition;
in vec3 fNormal;

out out_color;

void main()
{
    vec3 normalDir = normalize(fNormal);
    mat4 invView = inverse(viewMatrix);
    vec3 cameraPos = invView * vec4(0.0,0.0,0.0,1.0); // extract last column
    vec3 viewDir = normalize(vec3(cameraPos - fPosition));
    vec3 lightDir;
    float attenuation;

    vec4 matDiffuseColor = mat.diffuse;
    if (mat.hasTexture) {
        matDiffuseColor *= texture2D(diffuseTexture, fTexCoord);
    }
    vec3 totalLight = vec3(scene_ambient) * vec3(mat.ambient);

    for (int i = 0; i < nLights; i++) {
        if (lights[i].position.w == 0.0) { 
            attenuation = 1.0; // no attenuation
            lightDir = normalize(vec3(lights[i].position)); // directional light
        } else {
            vec3 positionToLight = vec3(lights[i].position - fPosition);
            float dist = length(positionToLight);
            lightDir = normalize(positionToLight); // point light/spotlight
            attenuation = 1.0 / (lights[i].constantAttenuation
                                + lights[i].linearAttenuation * dist
                                + lights[i].quadraticAttenuation * dist * dist);
            if (lights[i].spotCutoff <= 90.0) {
                // spotlight
                float spotCos = max(dot(-lighDir, normalize(lights[i].spotDirection)), 0.0);
                if (spotCos < cos(radians(lights[i].spotCutoff))) {
                    attenuation = 0.0;
                } else {
                    attenuation *= pow(spotCos, lights[i].spotExponent);
                }
            }
        }

        vec3 diffuse = attenuation * vec3(lights[i].diffuse) * vec3(matDiffuseColor) * max(dot(normalDir, lightDir), 0.0);
        vec3 specular;
        if (dot(normalDir, lightDir) < 0.0) {
            // light is on the wrong side
            specular = vec3(0.0);
        } else {
            specular = attenuation * vec3(lights[i].specular) * vec3(mat.specular);
            specular *= pow(max(0.0, dot(reflect(-lightDir, normalDir), viewDir)), mat.shininess);
        }

        totalLight += diffuse + specular;
    }

    out_color = vec4(totalLight, mat.transparency);
}
