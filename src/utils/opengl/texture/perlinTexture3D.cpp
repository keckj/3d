

#include "perlin.h"
#include "perlinTexture3D.h"

#include <cstring>


	PerlinTexture3D::PerlinTexture3D(unsigned int textureWidth, unsigned int textureHeight, unsigned int textureLength,
					double *seed,
					double startFrequency, double startAmp) :
		Texture3D(textureWidth, textureHeight, textureLength, 
		GL_RGBA8UI,
		NULL, 0, 0)	

	{
	
		memcpy(_seed, seed, 3*sizeof(double));

		for (int j = 0; j < 4; j++) {
			_frequency[j] = startFrequency * (1<<j);
			_amp[j] = startAmp / (1<<j);
		}

		make3DNoiseTexture();
	}
	
	PerlinTexture3D::PerlinTexture3D(unsigned int textureWidth, unsigned int textureHeight, unsigned int textureLength,
									 double *seed, 
									 double *frequency, double *amp) :
		Texture3D(textureWidth, textureHeight, textureLength, 
		GL_RGBA8UI,
		NULL, 0, 0)	
	{

		memcpy(_seed, seed, 3*sizeof(double));
		memcpy(_frequency, frequency, 4*sizeof(double));
		memcpy(_amp, amp, 4*sizeof(double));
		
		make3DNoiseTexture();
	}

PerlinTexture3D::~PerlinTexture3D() {
	delete [] (GLubyte*)_texels;
}

void PerlinTexture3D::make3DNoiseTexture(void)
{
    
    double ni[3];
	double inci, incj, inck;

	double frequency, amp;
    
	GLubyte *ptr;
	GLubyte *noise3DTexPtr;

    noise3DTexPtr = new GLubyte[4*_width *_height *_length];
		
	if(!noise3DTexPtr)
    {
         log_console.errorStream() << "[3D Perlin Texture Generation]  Could not allocate 3D noise texture !";
         exit(1);
    }
       

    for (unsigned int f = 0; f < 4; ++f)
    {
        ptr = noise3DTexPtr + f;

		ni[0] = _seed[0];
		ni[1] = _seed[1];
		ni[2] = _seed[2];

		amp = _amp[f];
		frequency = _frequency[f];

        inci = 1.0 / (_length / frequency);
        for (unsigned int i = 0; i < _length; ++i, ni[0] += inci)
        {
            incj = 1.0 / (_height / frequency);
            for (unsigned int j = 0; j < _height; ++j, ni[1] += incj)
            {
               inck = 1.0 / (_width / frequency);
               for (unsigned int k = 0; k < _width; ++k, ni[2] += inck, ptr+= 4)
                {
                   *ptr = (GLubyte)(((Perlin::noise3(ni)+1.0) * amp)*128.0);
                }
            }
        }
				 std::cout << std::endl;
    }
		
	setData(noise3DTexPtr, GL_RGBA, GL_UNSIGNED_BYTE);
}
