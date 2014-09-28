
#include "headers.h"
#include "marchingCubes.h"
#include "globals.h"
#include "mc_utils.h"
#include "perlinTexture3D.h"
#include "perlin.h"
#include "utils.h"

bool MarchingCubes::_init = false;
unsigned int MarchingCubes::_triTableUBO = 0;
unsigned int MarchingCubes::_lookupTableUBO = 0;
unsigned int MarchingCubes::_poissonDistributionsUBO = 0;

MarchingCubes::MarchingCubes(unsigned int width, unsigned int height, unsigned int length, float voxelSize) :
		_density(0), _normals_occlusion(0), _terrain_texture(0),
        _textureWidth(width), _textureHeight(height), _textureLength(length),
        _voxelGridWidth(width-1), _voxelGridHeight(height-1), _voxelGridLength(length-1), 
        _voxelWidth(voxelSize), _voxelHeight(voxelSize), _voxelLength(voxelSize),
        _drawProgram(0), _densityProgram(0), _normalOcclusionProgram(0), _marchingCubesProgram(0),
		_vertexVBO(0), _fullscreenQuadVBO(0), _marchingCubesLowerLeftXY_VBO(0),           
		_marchingCubesFeedbackVertexTBO(0), _nTriangles(0),
		_generalDataUBO(0)
{

        if(!_init) {
                generateUniformBlockBuffers();
        }

        //create general data uniform block
        GLfloat generalData[12] = {
                (GLfloat)_textureWidth,   (GLfloat)_textureHeight,   (GLfloat)_textureLength,   0.0f,
                (GLfloat)_voxelGridWidth, (GLfloat)_voxelGridHeight, (GLfloat)_voxelGridLength, 0.0f,
                (GLfloat)_voxelWidth,     (GLfloat)_voxelHeight,     (GLfloat)_voxelLength,     0.0f
        };


        glGenBuffers(1, &_generalDataUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, _generalDataUBO);
        glBufferData(GL_UNIFORM_BUFFER, 12*sizeof(GLfloat), generalData, GL_STATIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        //create textures
        _density = new Texture3D(_textureWidth, _textureHeight,_textureLength, GL_R16F, 0, GL_RED, GL_FLOAT);
        _density->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
        _density->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
        _density->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
        _density->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        _density->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        _density->bindAndApplyParameters(0); //allocate texture and apply parameters

        _normals_occlusion = new Texture3D(_textureWidth, _textureHeight,_textureLength, GL_RGBA32F, 0, GL_RGBA, GL_FLOAT);
        _normals_occlusion->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
        _normals_occlusion->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
        _normals_occlusion->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
        _normals_occlusion->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        _normals_occlusion->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        _normals_occlusion->bindAndApplyParameters(1); //allocate texture and apply parameters

		_terrain_texture = new Texture2D("textures/terrain/striation.png", "png");
        _terrain_texture->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_REPEAT));
        _terrain_texture->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_REPEAT));
        _terrain_texture->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_REPEAT));
        _terrain_texture->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        _terrain_texture->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));


        generateQuads();
        generateFullScreenQuad();
        generateMarchingCubesPoints();

        makeDensityProgram();
        makeNormalOcclusionProgram();
        makeMarchingCubesProgram();
        makeDrawProgram();

		computeDensitiesAndNormals();
		marchCubes();
}

MarchingCubes::~MarchingCubes() {
		Texture *textures[] = {_density, _normals_occlusion, _terrain_texture};
		Program *programs[] = {_drawProgram, _densityProgram, _normalOcclusionProgram, _marchingCubesProgram};
		unsigned int buffers[] = {_vertexVBO, _fullscreenQuadVBO, _marchingCubesLowerLeftXY_VBO, _marchingCubesFeedbackVertexTBO, _generalDataUBO};

		for (int i = 0; i < 3; i++) {
			delete textures[i];
		}
		
		for (int i = 0; i < 4; i++) {
			delete programs[i];
		}

		for (int i = 0; i < 5; i++) {
			if(glIsBuffer(buffers[i]))
				glDeleteBuffers(1, &buffers[i]);
		}
}

void MarchingCubes::drawDownwards(const float *currentTransformationMatrix) {
	_drawProgram->use();
        
	glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, _generalDataUBO);
	glUniformMatrix4fv(_drawUniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
	

	glBindBuffer(GL_ARRAY_BUFFER, _marchingCubesFeedbackVertexTBO);           
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	
    glDrawArrays(GL_TRIANGLES, 0, _nTriangles*3);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
}
		
void MarchingCubes::computeDensitiesAndNormals() {
        
		//The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
        GLuint frameBuffer = 0;
        glGenFramebuffers(1, &frameBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _density->getTextureId(), 0); //level 0 

        static const GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers); 

		Utils::checkFrameBufferStatus();

        // Render on the whole framebuffer, complete from the lower left corner to the upper right
        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0,0,_textureWidth,_textureHeight); 

        _densityProgram->use();
        glUniform1i(_densityUniformLocs["totalLayers"], _textureLength);
        glUniform2f(_densityUniformLocs["textureSize"], _textureWidth, _textureHeight);

        glBindBuffer(GL_ARRAY_BUFFER, _fullscreenQuadVBO);           
        glEnableVertexAttribArray(0);
        glVertexAttribDivisor(0,0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, _textureLength);

        //Render full screen triangle to generate normals and occlusion texture
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _normals_occlusion->getTextureId(), 0); 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		Utils::checkFrameBufferStatus();
        
		_normalOcclusionProgram->use();

        glBindBufferBase(GL_UNIFORM_BUFFER, 0 , _poissonDistributionsUBO);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1 , _generalDataUBO);

        glBindBuffer(GL_ARRAY_BUFFER, _fullscreenQuadVBO);           
        glEnableVertexAttribArray(0);
        glVertexAttribDivisor(0,0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, _textureLength);

        glBindBuffer(GL_ARRAY_BUFFER, 0);           
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(0);

        glPopAttrib();
}

void MarchingCubes::marchCubes() {
        unsigned int query[2];
        glGenQueries(2, query);

        _marchingCubesProgram->use();
		
		glEnable(GL_RASTERIZER_DISCARD);

        glBindBufferBase(GL_UNIFORM_BUFFER, 0 , _lookupTableUBO);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1 , _triTableUBO);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2 , _generalDataUBO);

        glBindBuffer(GL_ARRAY_BUFFER, _marchingCubesLowerLeftXY_VBO);           
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisor(0,0);
        glEnableVertexAttribArray(0);

        glBeginQuery(GL_PRIMITIVES_GENERATED, query[0]);
        glDrawArraysInstanced(GL_POINTS, 0, _voxelGridWidth*_voxelGridHeight, _voxelGridLength);
        glFlush();
        glEndQuery(GL_PRIMITIVES_GENERATED);

        unsigned int primitivesGenerated;
        glGetQueryObjectuiv(query[0], GL_QUERY_RESULT, &primitivesGenerated);

		if(glIsBuffer(_marchingCubesFeedbackVertexTBO))
			glDeleteBuffers(1, &_marchingCubesFeedbackVertexTBO);
        glGenBuffers(1, &_marchingCubesFeedbackVertexTBO);
        glBindBuffer(GL_ARRAY_BUFFER, _marchingCubesFeedbackVertexTBO);
        glBufferData(GL_ARRAY_BUFFER, primitivesGenerated*4*3*sizeof(GLfloat), 0, GL_STATIC_READ);

        const GLchar* feedbackVaryings[] = { "GS_FS_VERTEX.worldPos" };
        
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, _marchingCubesFeedbackVertexTBO);
        glTransformFeedbackVaryings(_marchingCubesProgram->getProgramId(), 1, feedbackVaryings, GL_SEPARATE_ATTRIBS);

		//update program
		_marchingCubesProgram->link();
		_marchingCubesProgram->use();
        
        glBeginQuery(GL_PRIMITIVES_GENERATED, query[0]);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query[1]);
        glBeginTransformFeedback(GL_TRIANGLES);
		glDrawArraysInstanced(GL_POINTS, 0, _voxelGridWidth*_voxelGridHeight, _voxelGridLength);
        glEndTransformFeedback();
        glFlush();
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        
        unsigned int primitivesWritten;
        glGetQueryObjectuiv(query[0], GL_QUERY_RESULT, &primitivesGenerated);
        glGetQueryObjectuiv(query[1], GL_QUERY_RESULT, &primitivesWritten);
        
		
		log_console.infoStream() << "[Marching Cube] Generated " <<  primitivesGenerated << " primitives.";
		log_console.infoStream() << "[Marching Cube] Wrote " <<  primitivesGenerated << " primitives.";
		_nTriangles = primitivesWritten;
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
		glDisable(GL_RASTERIZER_DISCARD);
        glTransformFeedbackVaryings(_marchingCubesProgram->getProgramId(), 0, 0, GL_SEPARATE_ATTRIBS);
        glUseProgram(0);

        glDeleteQueries(2, query);
}

void MarchingCubes::generateQuads() {

        float quads[4*3] = {0, 0, 0,
                0, 1, 0,
                1, 1, 0,
                1, 0, 0};

        glGenBuffers(1, &_vertexVBO);
        glBindBuffer(GL_ARRAY_BUFFER, _vertexVBO);
        glBufferData(GL_ARRAY_BUFFER, 4*3*sizeof(float), quads, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MarchingCubes::generateFullScreenQuad() {
        float buffer[] = {
                -1.0f, -1.0f, 0.0f,
                1.0f, -1.0f, 0.0f,
                1.0f, 1.0f, 0.0f,

                -1.0f, 1.0f, 0.0f,
                -1.0f, -1.0f, 0.0f,
                1.0f, 1.0f, 0.0f,
        };

        glGenBuffers(1, &_fullscreenQuadVBO);
        glBindBuffer(GL_ARRAY_BUFFER, _fullscreenQuadVBO);
        glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(float), buffer, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MarchingCubes::generateMarchingCubesPoints() {

        float *lowerLeftXY = new float[2*_voxelGridWidth*_voxelGridHeight];

        float stepX = 1.0f/_voxelGridWidth;
        float stepY = 1.0f/_voxelGridHeight;

        float posY = 0;
        for (unsigned int j = 0; j < _voxelGridHeight; j++) {
                float posX = 0;
                for (unsigned int i = 0; i < _voxelGridWidth; i++) {
                        lowerLeftXY[2*(j*_voxelGridWidth + i) + 0] = posX;
                        lowerLeftXY[2*(j*_voxelGridWidth + i) + 1] = posY;
                        posX += stepX;
                }
                posY += stepY;
        }

        glGenBuffers(1, &_marchingCubesLowerLeftXY_VBO);
        glBindBuffer(GL_ARRAY_BUFFER, _marchingCubesLowerLeftXY_VBO);
        glBufferData(GL_ARRAY_BUFFER, 2*_voxelGridWidth*_voxelGridHeight*sizeof(float), lowerLeftXY, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        delete [] lowerLeftXY;
}

void MarchingCubes::makeDrawProgram() {
        _drawProgram = new Program("MC Draw");
        _drawProgram->bindAttribLocation(0, "vertex_position");
        _drawProgram->bindFragDataLocation(0, "out_colour");
        _drawProgram->bindUniformBufferLocations("0 1", "projectionView generalData");

        _drawProgram->attachShader(Shader("shaders/marchingCubes/draw_vs.glsl", GL_VERTEX_SHADER));
        _drawProgram->attachShader(Shader("shaders/marchingCubes/draw_fs.glsl", GL_FRAGMENT_SHADER));

        _drawProgram->link();
        _drawUniformLocs = _drawProgram->getUniformLocationsMap("modelMatrix", true);
        
		Texture *tex[] = {_terrain_texture, _normals_occlusion};
		_drawProgram->bindTextures(tex, "terrain_texture normals_occlusion", false);
}

void MarchingCubes::makeDensityProgram() {
        _densityProgram = new Program("MC Density");
        _densityProgram->bindAttribLocations("0", "vertex_position");
        _densityProgram->bindFragDataLocation(0, "out_colour");

        _densityProgram->attachShader(Shader("shaders/marchingCubes/density_vs.glsl", GL_VERTEX_SHADER));
        _densityProgram->attachShader(Shader("shaders/marchingCubes/density_gs.glsl", GL_GEOMETRY_SHADER));
        _densityProgram->attachShader(Shader("shaders/marchingCubes/density_fs.glsl", GL_FRAGMENT_SHADER));

        _densityProgram->link();

        _densityUniformLocs = _densityProgram->getUniformLocationsMap("totalLayers textureSize", true);
}

void MarchingCubes::makeNormalOcclusionProgram() {
        _normalOcclusionProgram = new Program("MC Normal & Occlusion");
        _normalOcclusionProgram->bindAttribLocations("0", "vertex_position");
        _normalOcclusionProgram->bindFragDataLocation(0, "out_colour");
        _normalOcclusionProgram->bindUniformBufferLocations("0 1", "poissonDistributions generalData");

        _normalOcclusionProgram->attachShader(Shader("shaders/marchingCubes/normals_vs.glsl", GL_VERTEX_SHADER));
        _normalOcclusionProgram->attachShader(Shader("shaders/marchingCubes/normals_gs.glsl", GL_GEOMETRY_SHADER));
        _normalOcclusionProgram->attachShader(Shader("shaders/marchingCubes/normals_fs.glsl", GL_FRAGMENT_SHADER));

        _normalOcclusionProgram->link();

        _normalOcclusionProgram->bindTextures(&_density, "density", true);
}

void MarchingCubes::makeMarchingCubesProgram() {
        _marchingCubesProgram = new Program("MC Marching Cube");
        _marchingCubesProgram->bindAttribLocations("0", "voxelLowerLeftXY");
        _marchingCubesProgram->bindUniformBufferLocations("0 1 2", "lookupTable triangleTable generalData");

        _marchingCubesProgram->attachShader(Shader("shaders/marchingCubes/marchingCube_vs.glsl", GL_VERTEX_SHADER));
        _marchingCubesProgram->attachShader(Shader("shaders/marchingCubes/marchingCube_gs.glsl", GL_GEOMETRY_SHADER));

        _marchingCubesProgram->link();

        _marchingCubesProgram->bindTextures(&_density, "density");
}

void MarchingCubes::generateUniformBlockBuffers() {
        log_console.infoStream() << "Size of GLbyte : " << sizeof(GLbyte);
        log_console.infoStream() << "Size of GLfloat : " << sizeof(GLfloat);
        log_console.infoStream() << "Size of GLint : " << sizeof(GLint);
        log_console.infoStream() << "Size of GLuint : " << sizeof(GLuint);
        log_console.infoStream() << "Generating static marching cubes uniform block buffers !";

        glGenBuffers(1, &_triTableUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, _triTableUBO);
        glBufferData(GL_UNIFORM_BUFFER, 5120*sizeof(GLfloat), MarchingCube::triangleTable, GL_STATIC_DRAW);

        glGenBuffers(1, &_lookupTableUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, _lookupTableUBO);

        glBufferData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 48*6*sizeof(GLfloat), 0, GL_STATIC_DRAW);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, 1024*sizeof(GLuint), MarchingCube::caseToNumPoly);
        glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 0*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::edgeStart);
        glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 1*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::edgeDir);
        glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 2*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskA0123);
        glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 3*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskB0123);
        glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 4*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskA4567);
        glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 5*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskB4567);

        glGenBuffers(1, &_poissonDistributionsUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, _poissonDistributionsUBO);

        glBufferData(GL_UNIFORM_BUFFER,    (256+128+64+32)*4*sizeof(GLfloat), 0, GL_STATIC_DRAW);
        glBufferSubData(GL_UNIFORM_BUFFER, (  0+  0+ 0)*4*sizeof(GLfloat), 256*4*sizeof(GLfloat), MarchingCube::poissonRayDirs_256);
        glBufferSubData(GL_UNIFORM_BUFFER, (256+  0+ 0)*4*sizeof(GLfloat), 128*4*sizeof(GLfloat), MarchingCube::poissonRayDirs_128);
        glBufferSubData(GL_UNIFORM_BUFFER, (256+128+ 0)*4*sizeof(GLfloat),  64*4*sizeof(GLfloat), MarchingCube::poissonRayDirs_64);
        glBufferSubData(GL_UNIFORM_BUFFER, (256+128+64)*4*sizeof(GLfloat),  32*4*sizeof(GLfloat), MarchingCube::poissonRayDirs_32);

        glBindBuffer(GL_UNIFORM_BUFFER, 0);
}
