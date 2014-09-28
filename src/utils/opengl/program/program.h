
#ifndef PROGRAM_H
#define PROGRAM_H

#include "shader.h"
#include "texture.h"
#include <string>
#include <vector>
#include <map>

class Program {
	
	public:
		Program(std::string const &name); 
		~Program(); 
		
		void attachShader(Shader const &shader);

		void bindAttribLocation(unsigned int location, std::string const &attribVarName);
		void bindFragDataLocation(unsigned int location, std::string const &fragVarName);
		void bindUniformBufferLocation(unsigned int location, std::string const &blockName);
		
		void bindAttribLocations(std::string locations, std::string const &attribVarNames);
		void bindFragDataLocations(std::string locations, std::string const &fragVarNames);
		void bindUniformBufferLocations(std::string locations, std::string const &blockNames);

		void link(); //can be called multiple times (ex : transform feedback)
		void use() const; //use program, request linked textures, update linked texture uniforms
		
		unsigned int getProgramId() const;

		//assert check if the uniform really exists
		const std::vector<int> getUniformLocations(std::string const &varNames, bool assert = false); //uniform var names separated by space
		const std::map<std::string,int> getUniformLocationsMap(std::string const &varNames, bool assert = false); //separated by space

		void bindTextures(Texture **textures, std::string uniformNames, bool assert = false);
		
		static void resetDefaultGlProgramState(); // for debugging purpose only

	private:
		std::string programName;
		unsigned int programId;

		bool linked;

		std::map<std::string, unsigned int> attribLocations;
		std::map<std::string, unsigned int> uniformBufferLocations;
		std::vector<std::pair<int, Texture *> > linkedTextures;
		std::string logProgramHead;

		void bindUniformBlocks(bool assert);
};


#endif /* end of include guard: PROGRAM_H */
