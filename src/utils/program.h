
#ifndef PROGRAM_H
#define PROGRAM_H

#include "shader.h"
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
		
		void bindAttribLocations(std::string locations, std::string const &attribVarNames);
		void bindFragDataLocations(std::string locations, std::string const &fragVarNames);

		void link(); //can be called multiple times (transform feedback)
		void use() const; 
		
		unsigned int getProgramId() const;

		const std::vector<int> getUniformLocations(std::string const &varNames); //uniform var names separated by space
		const std::vector<int> getUniformLocationsAndAssert(std::string const &varNames); //assert if they exist

		const std::map<std::string,int> getUniformLocationsMap(std::string const &varNames); //separated by space
		const std::map<std::string,int> getUniformLocationsMapAndAssert(std::string const &varNames);
		
		static void resetDefaultGlProgramState(); //TODO

	private:
		std::string programName;
		unsigned int programId;

		bool linked;

		std::map<std::string, unsigned int> attribLocations;
		std::string logProgramHead;
};


#endif /* end of include guard: PROGRAM_H */
