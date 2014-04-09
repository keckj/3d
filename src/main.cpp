
#include <GL/glew.h>

#include <qapplication.h>
#include "viewer.h"

#include "objLoader/ObjLoader.h"

#include "waves.h"
#include "fog.h"

#include "terrain.h"
#include "shader.h"
#include "SeaDiver.h"
#include "Rectangle.h"
#include <QWidget>

#include <ostream>
#include <cassert>

#include "log.h"
#include "program.h"
#include "globals.h"
#include "cube.h"
#include "cudaUtils.h"
#include "texture.h"


using namespace std;
using namespace log4cpp;

int main(int argc, char** argv) {


        srand(time(NULL));


        log4cpp::initLogs();

        CudaUtils::logCudaDevices(log_console);

        log_console.infoStream() << "[Rand Init] ";
        log_console.infoStream() << "[Logs Init] ";

        // glut initialisation (mandatory) 
        glutInit(&argc, argv);
        log_console.infoStream() << "[Glut Init] ";

        // Read command lines arguments.
        QApplication application(argc,argv);
        log_console.infoStream() << "[Qt Init] ";

        // Instantiate the viewer.
        Viewer viewer;
        viewer.setWindowTitle("Sea diver");
        viewer.show();

        //glew initialisation (mandatory)
        log_console.infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

        Globals::init();
        Globals::print(std::cout);
        Globals::check();
		
		Texture::init();

        log_console.infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";


        // SIMPLE EXAMPLE FOR NOOBS //
        Program program("_SwagDePoulpe_");

        program.bindAttribLocation(0, "vertex_position");
        program.bindAttribLocations("1 10", "vertex_position vertex_colour");
        program.bindFragDataLocation(0, "out_colour");

        program.attachShader(Shader("shaders/common/vs.glsl", GL_VERTEX_SHADER));
        program.attachShader(Shader("shaders/common/fs.glsl", GL_FRAGMENT_SHADER));

        program.link();

        const std::vector<int> uniforms_vec = program.getUniformLocations("modelMatrix projectionMatrix viewMatrix");
        const std::map<std::string,int> uniforms_map = program.getUniformLocationsMap("modelMatrix poulpy");
        //std::vector<int> uniforms_vec = program.getUniformLocationsAndAssert("modelMatrix projectionMatrix viewMatrix");
        //std::map<std::string,int> uniforms_map = program.getUniformLocationsMapAndAssert("modelMatrix projectionMatrix viewMatrix");

        //program.use();
        ///////////////////////////////////////////////

		
		Texture texture("textures/dirt 1.png","png",GL_TEXTURE_2D);
		texture.addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_REPEAT));
		texture.addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_REPEAT));
		texture.addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		texture.generateMipMap();
		texture.bindAndApplyParameters(0);

		Texture texture2("textures/dirt 3.png", "png", GL_TEXTURE_2D);
		texture2.addParameters(texture.getParameters());
		texture2.addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		texture2.bindAndApplyParameters(1);

		std::vector<unsigned int> id = Texture::requestTextures(100);


        /*

        // -- shaders --
        Shader *fs = new Shader("shaders/vertex_shader.glsl", GL_VERTEX_SHADER);
        Shader *vs = new Shader("shaders/fragment_shader.glsl", GL_FRAGMENT_SHADER);

        // -- programme --
        unsigned int shader_program = glCreateProgram ();
        glAttachShader (shader_program, fs->getShader());
        glAttachShader (shader_program, vs->getShader());

        // -- location des attributs
        glBindAttribLocation(shader_program, 0, "vertex_position");
        glBindAttribLocation(shader_program, 1, "vertex_colour");
        glBindAttribLocation(shader_program, 2, "vertex_normal");
        //glBindFragDataLocation(shader_program, 0, "out_colour");

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
        << glGetAttribLocation(shader_program, "vertex_normal") << "\t";
        //<< glGetFragDataLocation(shader_program, "out_colour");

        assert(glGetAttribLocation(shader_program, "vertex_position")==0);
        assert(glGetAttribLocation(shader_program, "vertex_colour")!=-1);
        assert(glGetAttribLocation(shader_program, "vertex_normal")==-1);
        //assert(glGetFragDataLocation(shader_program, "out_colour")==0);

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

        QImage text1 = QGLWidget::convertToGLFormat(QImage("textures/forest 13.png","png"));
        QImage text2 = QGLWidget::convertToGLFormat(QImage("textures/grass 9.png","png"));
        QImage text3 = QGLWidget::convertToGLFormat(QImage("textures/grass 7.png","png"));
        QImage text4 = QGLWidget::convertToGLFormat(QImage("textures/dirt 4.png","png"));
        QImage text5 = QGLWidget::convertToGLFormat(QImage("textures/snow 1.png","png"));

        assert(text1.bits());
        assert(text2.bits());
        assert(text3.bits());
        assert(text4.bits());
        assert(text5.bits());

        unsigned int *textures = new unsigned int[5];
        glGenTextures(5, textures);

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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                        text1.width(), text1.height(), 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, text1.bits());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textures[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                        text2.width(), text2.height(), 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, text2.bits());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, textures[2]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                        text3.width(), text3.height(), 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, text3.bits());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, textures[3]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                        text4.width(), text4.height(), 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, text4.bits());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, textures[4]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                        text5.width(), text5.height(), 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, text5.bits());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);

        QImage rgb_heightmap = QGLWidget::convertToGLFormat(QImage("img/tamriel3.jpg","jpg"));
        assert(rgb_heightmap.bits());
        unsigned char *black_img = new unsigned char[rgb_heightmap.height()*rgb_heightmap.width()];

        for (int i = 0; i < rgb_heightmap.width(); i++) {
                for (int j = 0; j < rgb_heightmap.height(); j++) {
                        QRgb color = rgb_heightmap.pixel(i,j);
                        black_img[j*rgb_heightmap.width() + i] = (unsigned char) ((qRed(color) + qGreen(color) + qBlue(color))/3);
                }
        }
        viewer.addRenderable(new Terrain(black_img, rgb_heightmap.width(),rgb_heightmap.height(), true, shader_program, modelMatrixLocation, projectionMatrixLocation, viewMatrixLocation));
        */
                // build your scene here
                //
                //
                //glDisable(GL_LIGHTING);
                //glDisable(GL_TEXTURE_2D);


        viewer.setSceneRadius(100.0f);
	/* viewer.addRenderable(new Terrain(black_img, rgb_heightmap.width(),rgb_heightmap.height(), true, shader_program, modelMatrixLocation, projectionMatrixLocation, viewMatrixLocation)); */

        viewer.addRenderable(new Cube());

        // Run main loop.
        return application.exec();
}

