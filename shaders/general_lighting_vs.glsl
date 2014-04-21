#version 330 core

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix = transpose(inverse(mat3(viewMatrix * modelMatrix)));

in vec4 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;

out vec4 fPosition;
out vec3 fNormal;
out vec2 fTexCoord;

void main()
{
    fNormal = normalize(normalMatrix * vertexNormal);
    fTexCoord = vertexTexCoord;
    fPosition = viewMatrix * modelMatrix * vertexPosition;
    gl_Position = projectionMatrix * fPosition;
}
