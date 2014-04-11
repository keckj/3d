
#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>
#include <map>
#include <list>
#include <QImage>

#include "parameter.h"

class Texture {
	
	public:

		void addParameter(Parameter param);
		void addParameters(const std::list<Parameter> &paramList);
		void generateMipMap();

		const std::list<Parameter> getParameters() const;

		unsigned int getTextureId() const;
		int getLastKnownLocation() const;

		bool isBinded() const; //check wether the texture is still linked to its last known location or not
		
		virtual void bindAndApplyParameters(unsigned int location) = 0;
	
		static void init();
		static std::vector<unsigned int> requestTextures(unsigned int nbRequested);
		static void sortHitMap();
		static void reportHitMap();

	protected:
		Texture(GLenum textureType);
		~Texture();

		GLenum textureType;

		unsigned int textureId;
		int lastKnownLocation;

		std::list<Parameter> params;
				
		std::string logTextureHead;

		bool mipmap;

		//texture must be linked
		void applyParameters() const;
	
		static bool _init;
		static std::vector<int> textureLocations;
		static std::map<unsigned int, long> locationsHitMap;
		static std::vector<std::pair<long, unsigned int> > reversedHitMap; 

		static bool compareFunc(std::pair<long, unsigned int> a, std::pair<long, unsigned int> b);
};

#endif /* end of include guard: TEXTURE_H */
