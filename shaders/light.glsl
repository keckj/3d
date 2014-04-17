///////////////////////////
///         WIP         ///
///////////////////////////


// NOTE: Possibilité de #if charette pour illuminer en calculant sur les vertex au lieu des fragments


/***********************************************************************************/
// Vertex Shader 
/***********************************************************************************/


in vec3 position;
in vec3 normal;
uniform mat3 normalMatrix; // transpose(inverse(mat3(viewMatrix * modelMatrix))) [ou juste mat3(viewMatrix * modelMatrix) si scaling uniforme dans toutes les directions]
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
out vec3 fNormal;
out vec3 fPosition;
out vec3 l_dir;


void main()
{
  fNormal = normalize(normalMatrix * normal);
  vec4 pos = viewMatrix * modelMatrix * vec4(position, 1.0);
  fPosition = pos.xyz;
  l_dir = (viewMatrix * modelMatrix * vec4(5.0,10.0,1.0,1.0)).xyz;
  gl_Position = projectionMatrix * pos;
}


/***********************************************************************************/
// Fragment Shader 
/***********************************************************************************/


in vec3 fPosition;
in vec3 fNormal;

struct lightSource
{
  vec4 position;
  vec4 diffuse;
  vec4 specular;
  float shininess;
  float constantAttenuation, linearAttenuation, quadraticAttenuation;
};
in lightSource light0;
//in lightSource light1; possible avec une boucle à borne fixe


void main()
{
  vec3 l = normalize(light0.position);
  /*vec4 specular = vec4(0.05,0.05,0.05,1.0);
  vec4 diffuse = vec4(57.0/256.0,88.0/256.0,121.0/256.0,1.0);
  vec4 ambient = diffuse/4.0;
   float shininess = 10.0;?
  EX WATER
  */
  
    // set the specular term to black
    vec4 spec = vec4(0.0);
 
    // normalize both input vectors
    vec3 n = normalize(fNormal);
    vec3 e = normalize(-fPosition);
 
    float lambertTerm  = max(dot(n,l), 0.0);
 
    // if the vertex is lit compute the specular color
    /*if (lambertTerm  > 0.0) {
        // compute the half vector
        vec3 h = normalize(l + e); 
        // compute the specular term into spec
        float intSpec = max(dot(h,n), 0.0);*/
        float intSpec = max(0.0, dot(reflect(-l, n), e));
        spec = specular * pow(intSpec,shininess);
    }
    
    gl_FragColor = max(lambertTerm  *  diffuse + spec, ambient);
}