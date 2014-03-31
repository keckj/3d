
#include <GL/glew.h>

#include <qapplication.h>
#include "viewer.h"

#include "objLoader/ObjLoader.h"

#include <ctime>
#include "waves.h"
#include "fog.h"

#include "terrain.h"
#include "shader.h"
#include "image.h"
#include "SeaDiver.h"

#include <ostream>
#include <opencv2/core/core.hpp>
#include <cassert>

#include "log.h"


using namespace std;
using namespace cv;
using namespace log4cpp;

int main(int argc, char** argv) {

    srand(time(NULL));
	log4cpp::initLogs();
    

    // Read command lines arguments.
    QApplication application(argc,argv);
	log_console.infoStream() << "[Qt Init] ";
    
	// Instantiate the viewer.
    Viewer viewer;
    viewer.setWindowTitle("Sea diver");
	
	// glut initialisation (mandatory) 
	glutInit(&argc, argv);
	log_console.infoStream() << "[Glut Init] ";
	
	// glew initialisation (mandatory)
	//log_console.infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

	int defaultProgramm;
	glGetIntegerv(GL_CURRENT_PROGRAM, &defaultProgramm);
	log_console.infoStream() << "Current programm is " << defaultProgramm;


	
	/*	
	// -- shaders --
	Shader *vs = new Shader("shaders/vertex_shader.glsl", GL_VERTEX_SHADER);
	Shader *fs = new Shader("shaders/fragment_shader.glsl", GL_FRAGMENT_SHADER);
	
	// -- programme --
	unsigned int shader_program = glCreateProgram ();
	glAttachShader (shader_program, fs->getShader());
	glAttachShader (shader_program, vs->getShader());
	
	// -- location des attributs
	glBindAttribLocation(shader_program, 0, "vertex_position");
	glBindAttribLocation(shader_program, 1, "vertex_colour");
	glBindAttribLocation(shader_program, 2, "vertex_normal");
	glBindFragDataLocation(shader_program, 0, "out_colour");

	// -- link du programme
	glLinkProgram(shader_program);
	int status;
  	glGetProgramiv(shader_program, GL_LINK_STATUS, &status);
	assert(status);
	
	log_console.infoStream() << "Link shader program OK";
	log_console.infoStream() << "ID shader program = " << shader_program;
	
	// -- post link verifications
	log_console.infoStream() << "Updated locations \t"
	<< glGetAttribLocation(shader_program, "vertex_position") << "\t"
	<< glGetAttribLocation(shader_program, "vertex_colour") << "\t"
	<< glGetAttribLocation(shader_program, "vertex_normal") << "\t"
	<< glGetFragDataLocation(shader_program, "out_colour");
	
	assert(glGetAttribLocation(shader_program, "vertex_position")==0);
	assert(glGetAttribLocation(shader_program, "vertex_colour")!=-1);
	assert(glGetAttribLocation(shader_program, "vertex_normal")==-1);
	assert(glGetFragDataLocation(shader_program, "out_colour")==0);
	
	// -- variables uniformes --
	int modelMatrixLocation = glGetUniformLocation(shader_program, "modelMatrix");
	int viewMatrixLocation = glGetUniformLocation(shader_program, "viewMatrix");
	int projectionMatrixLocation = glGetUniformLocation(shader_program, "projectionMatrix");
	int texture1Location = glGetUniformLocation(shader_program, "texture_1");
	int texture2Location = glGetUniformLocation(shader_program, "texture_2");
	int texture3Location = glGetUniformLocation(shader_program, "texture_3");
	int texture4Location = glGetUniformLocation(shader_program, "texture_4");
	int texture5Location = glGetUniformLocation(shader_program, "texture_5");

	assert(modelMatrixLocation != -1);
	assert(viewMatrixLocation != -1);
	assert(projectionMatrixLocation != -1);
	assert(texture1Location != -1);
	assert(texture2Location != -1);
	assert(texture3Location != -1);
	assert(texture4Location != -1);
	assert(texture5Location != -1);

	log_console.infoStream() << "Uniform locations \t"
	<< modelMatrixLocation << "\t"
	<< viewMatrixLocation << "\t"
	<< projectionMatrixLocation << "\t"
	<< texture1Location << "\t"
	<< texture2Location << "\t"
	<< texture3Location << "\t"
	<< texture4Location << "\t"
	<< texture5Location;

	// -- textures --
	//glEnable(GL_TEXTURE_2D);

	cv::Mat text1 = imread("textures/forest 13.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text2 = imread("textures/grass 9.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text3 = imread("textures/grass 7.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text4 = imread("textures/dirt 4.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text5 = imread("textures/snow 1.png", CV_LOAD_IMAGE_COLOR);

	assert(text1.data);
	assert(text2.data);
	assert(text3.data);
	assert(text4.data);
	assert(text5.data);
	cv::Mat reversedText1 = text1.clone();
	cv::Mat reversedText2 = text2.clone();
	cv::Mat reversedText3 = text3.clone();
	cv::Mat reversedText4 = text4.clone();
	cv::Mat reversedText5 = text5.clone();
	Image::reverseImage(text1, reversedText1);
	Image::reverseImage(text2, reversedText2);
	Image::reverseImage(text3, reversedText3);
	Image::reverseImage(text4, reversedText4);
	Image::reverseImage(text5, reversedText5);

	unsigned int *textures = new unsigned int[5];
	glGenTextures(3, textures);
	
	// assign texture units
	glUseProgram(shader_program);
	glUniform1i(texture1Location, 0);
	glUniform1i(texture2Location, 1);
	glUniform1i(texture3Location, 2);
	glUniform1i(texture4Location, 3);
	glUniform1i(texture5Location, 4);
	glUseProgram(0);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
			reversedText1.cols, reversedText1.rows, 0,
			GL_BGR, GL_UNSIGNED_BYTE, reversedText1.data);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, textures[1]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
			reversedText2.cols, reversedText2.rows, 0,
			GL_BGR, GL_UNSIGNED_BYTE, reversedText2.data);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, textures[2]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
			reversedText3.cols, reversedText3.rows, 0,
			GL_BGR, GL_UNSIGNED_BYTE, reversedText3.data);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, textures[3]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
			reversedText4.cols, reversedText4.rows, 0,
			GL_BGR, GL_UNSIGNED_BYTE, reversedText4.data);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, textures[4]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
			reversedText5.cols, reversedText5.rows, 0,
			GL_BGR, GL_UNSIGNED_BYTE, reversedText5.data);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	
	cv::Mat heightmap = imread("img/tamriel3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	assert(heightmap.data);
	cv::Mat reversedheightmap = heightmap.clone();
	Image::reverseImage(heightmap, reversedheightmap);
	*/

    // build your scene here
	//viewer.addRenderable(new Terrain(reversedheightmap.data, reversedheightmap.cols,reversedheightmap.rows, true, shader_program, modelMatrixLocation));
	viewer.addRenderable(new SeaDiver());	
    viewer.show();

    // Run main loop.
    return application.exec();
}

