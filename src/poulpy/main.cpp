#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>

#define _USE_MATH_DEFINES 
#define ONE_DEG_IN_RAD 0.0174532925

#include "box.hpp"
#include "log.hpp"
#include "window.hpp"
#include "shader.hpp"
#include "utils.hpp"
#include "image.hpp"
#include "data.hpp"
#include "camera.hpp"
#include "renderTree.hpp"
#include "rotation.hpp"
#include "terrain.hpp"

#define USE_ROW_MAJOR GL_TRUE

using namespace std;

int g_gl_width;
int g_gl_height;
bool resize = false;

void glfw_error_callback (int error, const char* description) {
	log_console.errorStream() << "\nGLFW Error\t" << error << " : " << "\n\t" << description << "\n"; 
}


void glfw_window_size_callback (GLFWwindow* window, int width, int height) {
  g_gl_width = width;
  g_gl_height = height;
  resize = true;
}

double _update_fps_counter (GLFWwindow* window) {
	static double previous_seconds = glfwGetTime ();
	static int frame_count;
	double current_seconds = glfwGetTime ();
	double elapsed_seconds = current_seconds - previous_seconds;
	if (elapsed_seconds > 0.25) {
		previous_seconds = current_seconds;
		double fps = (double)frame_count / elapsed_seconds;
		char tmp[128];
		sprintf (tmp, "opengl @ fps: %.2lf", fps);
		glfwSetWindowTitle (window, tmp);
		frame_count = 0;
	}
	frame_count++;

	return elapsed_seconds;
}

int main( int argc, const char* argv[] )
{

	initLogs();

	glfwSetErrorCallback (glfw_error_callback);

	log_console.infoStream() << "\nStarting GLFW version " << glfwGetVersionString();

	if (!glfwInit()) {
		log_console.errorStream() <<  "ERROR: could not start GLFW3\n";
		return 1;
	} 

	//-- screen params --
	glfwWindowHint (GLFW_SAMPLES, 4);

	GLFWmonitor* mon = glfwGetPrimaryMonitor ();
	const GLFWvidmode* vmode = glfwGetVideoMode (mon);

	Window *wd = new Window(vmode->width, vmode->height ,"test");
	g_gl_width = vmode->width;
	g_gl_height = vmode->height;
	resize = true;

	//-- callbacks --
	glfwSetWindowSizeCallback(wd->getWindow(), glfw_window_size_callback);
	
	log_gl_params();

	
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

	cv::Mat text1 = imread("../textures/forest 13.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text2 = imread("../textures/grass 9.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text3 = imread("../textures/grass 7.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text4 = imread("../textures/dirt 4.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat text5 = imread("../textures/snow 1.png", CV_LOAD_IMAGE_COLOR);

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

	// -- camera --
	float camSpeed = 10.0f;
	float camYawSpeed = 1.0f;
	vec3 initialCameraPosition = {0.0f, 0.0f, +50.0f};
	float eulerAngle[] = {0.0f, 0.0f, 0.0f};
	Rotation initialCameraRotation(
			RotationType::INTRINSIC_ROTATION, 
			OrientationType::EULER_ORIENTATION, 
			EulerOrder::ORDER_XYX,
			eulerAngle);

	Camera camera(0.1f, 100.0f, 67.0f*consts::oneDegInRad, 1680.0f/1050.0f,
			initialCameraPosition, initialCameraRotation,
			ALL_AXES);

	// -- initialisation des matrices de projection et de vue --
	glUseProgram(shader_program);
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_TRUE,
						camera.getViewMatrix());
	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_TRUE,
						camera.getProjectionMatrix());
	glUseProgram(0);
	

	//====================
	//=debut du programme=
	//====================
	
	assert(glGetError()==GL_NO_ERROR);
		
		
	Box box(1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,true, shader_program, modelMatrixLocation);	
	RenderTree tree(&box, true);

	//cv::Mat image = imread("img/heightmap.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat image = imread("img/tamriel3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat reversedImage = image.clone();
	Image::reverseImage(image, reversedImage);

	Terrain terrain((unsigned char *) reversedImage.data, reversedImage.cols,reversedImage.rows, true, modelMatrixLocation, shader_program);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);

	double elapsed_seconds;
	while (!glfwWindowShouldClose(wd->getWindow())) {
		elapsed_seconds = _update_fps_counter (wd->getWindow());

		// wipe the drawing surface clear
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glViewport (0, 0, g_gl_width, g_gl_height);
		
		terrain.draw(terrain.getRelativeModelMatrix());

		// put the stuff we've been drawing onto the display
		glfwSwapBuffers (wd->getWindow());

		// update other events like input handling 
		glfwPollEvents ();

		if (GLFW_PRESS == glfwGetKey(wd->getWindow(), GLFW_KEY_ENTER)) {
			glfwSetWindowShouldClose (wd->getWindow(), 1);
		}
		// control keys
		bool cam_moved = false;
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_X)) {
			camera.rotate(camYawSpeed * elapsed_seconds, 0,0);
			cam_moved = true;
		}
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_Y)) {
			camera.rotate(0, camYawSpeed * elapsed_seconds, 0);
			cam_moved = true;
		}
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_Z)) {
			camera.rotate(0, 0, camYawSpeed * elapsed_seconds);
			cam_moved = true;
		}
		
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_LEFT)) {
			camera.translate(-camSpeed * elapsed_seconds, 0, 0);
			cam_moved = true;
		}
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_RIGHT)) {
			camera.translate(+camSpeed * elapsed_seconds, 0, 0);
			cam_moved = true;
		}
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_UP)) {
			camera.translate(0, +camSpeed * elapsed_seconds, 0);
			cam_moved = true;
		}
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_DOWN)) {
			camera.translate(0, -camSpeed * elapsed_seconds, 0);
			cam_moved = true;
		}
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_PAGE_UP)) {
			camera.translate(0, 0, +camSpeed * elapsed_seconds);
			cam_moved = true;
		}
		if (glfwGetKey (wd->getWindow(), GLFW_KEY_PAGE_DOWN)) {
			camera.translate(0, 0, -camSpeed * elapsed_seconds);
			cam_moved = true;
		}

		// update view matrix
		glUseProgram(shader_program);
		if (cam_moved) {
			glUniformMatrix4fv (viewMatrixLocation, 1, USE_ROW_MAJOR, camera.getViewMatrix());
		}
		
		if(resize) {
			camera.setAspectRatio((float) g_gl_width / (float) g_gl_height);
			glUniformMatrix4fv (projectionMatrixLocation, 1, USE_ROW_MAJOR, camera.getProjectionMatrix());
			resize = false;
		}
		glUseProgram(0);

	}

	return 0;
}

