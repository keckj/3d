
#ifndef PERLINTEXTURE3D_H
#define PERLINTEXTURE3D_H

#include "texture3D.h"

class PerlinTexture3D : public Texture3D {

	public:
		//RGBA 8bit uint per channel texture, each layer j has an 'octave' of frequency startFrequency*2^j 
		//and an amplitude of startAmp / 2^j
		//j is between 0 and 3
		//RGBA layout :
		//R = high amplitude, low freq   GB    A = high freq, low amplitude
		//Texture width, height and length should be power of two
		//Seed is a vec of 3 doubles
		explicit PerlinTexture3D(unsigned int textureWidth, unsigned int textureHeight, unsigned int textureLength,
				double *seed,
				double startFrequency = 4.0f, double startAmp = 0.5f);

		//RGBA 8bit uint per channel texture, each layer j has an 'octave' of frequency startFrequency/2^j 
		//and an amplitude of startAmp * 2^j
		//j is between 0 and 3
		//RGBA layout :
		//R = first fres, first amp  GB  A = last freq, last amp
		//Seed is a vec3, freq and amp vec4
		//Texture width, height and length should be power of two
		explicit PerlinTexture3D(
				unsigned int textureWidth, unsigned int textureHeight, unsigned int textureLength,
				double *seed, 
				double *frequency, double *amp);

		~PerlinTexture3D();

	private:

		void make3DNoiseTexture();

		double _seed[3];
		double _amp[4]; 
		double _frequency[4];

};


#endif /* end of include guard: PERLINTEXTURE3D_H */
