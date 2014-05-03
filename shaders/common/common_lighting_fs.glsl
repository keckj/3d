#version 330

uniform mat4 viewMatrix;
uniform mat4 viewMatrixInv; //= inverse(viewMatrix);

const int nLights = 5;
struct Light {
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    vec4 spotDirection;
    float constantAttenuation, linearAttenuation, quadraticAttenuation;
    float spotCutoff, spotExponent;
    int isEnabled;
};// lights[nLights];

layout(std140) uniform LightBuffer {
    Light lights[nLights];
};

layout(std140) uniform Material {
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
    float transparency;
    int hasTexture;
} mat;

uniform vec4 scene_ambient = vec4(0.3,0.3,0.3,1.0);

uniform sampler2D diffuseTexture;

in vec4 fPosition;
in vec3 fNormal;
in vec2 fTexCoord;

in vec3 fogColor;
in float fogFactor;

out vec4 out_color;


vec4 applyFog(in vec4 fragColor) {
    return mix( vec4(fogColor,1.0), fragColor, fogFactor );
}


void main()
{
    vec3 normalDir = normalize(fNormal);
    //mat4 invView = inverse(viewMatrix);
    //vec4 cameraPos = invView * vec4(0.0,0.0,0.0,1.0); // extract last column
    //vec4 cameraPos = viewMatrixInv[3];
    vec4 cameraPos = -viewMatrix[3] * viewMatrix;
    vec3 viewDir = normalize(vec3(cameraPos - fPosition));
    vec3 lightDir;
    float attenuation;

    vec4 matDiffuseColor = mat.diffuse;
    if (mat.hasTexture != 0.0) {
        matDiffuseColor *= texture2D(diffuseTexture, fTexCoord);
    }
    
    vec3 totalLight = vec3(scene_ambient);// * vec3(mat.ambient);

    for (int i = 0; i < nLights; i++) {
        if (lights[i].isEnabled != 0.0) {
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
                    float spotCos = max(dot(-lightDir, normalize(vec3(lights[i].spotDirection))), 0.0);
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
                specular *= pow(max(0.0, dot(reflect(-lightDir, normalDir), viewDir)), max(mat.shininess,0.000001)); // pow(x,0) = NaN :(
            }

            totalLight += diffuse + specular;
        }
    }

    out_color = vec4(totalLight, mat.transparency);

    // apply fog
    out_color = applyFog(out_color);
}
