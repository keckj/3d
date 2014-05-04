

#include "seaweedGroup.h"
#include "globals.h"
#include "log.h"
#include "rand.h"

#include "constantForce.h"
#include "seaFlow.h"
#include "dynamicScheme.h"
#include "springsSystem.h"

SeeweedGroup::SeeweedGroup(unsigned int maxSeeweeds, unsigned int maxSubdivisions, float seeweedWidth) :
	ParticleGroup(maxSeeweeds*(1+maxSubdivisions), maxSeeweeds*maxSubdivisions), 
	_maxSeeweeds(maxSeeweeds), _maxSubdivisions(maxSubdivisions),
	_seeweedWidth(seeweedWidth)
{
	makeSeeweedsProgram();

	this->addKernel(new ConstantForce(0.02*qglviewer::Vec(0,9.81,0)));
	this->addKernel(new SeaFlow(qglviewer::Vec(1,0,0), 0.2, 0.01));
	this->addKernel(new SpringsSystem(true));
	this->addKernel(new DynamicScheme());
}

SeeweedGroup::~SeeweedGroup() {
	delete _seeweedsProgram;
}

void SeeweedGroup::spawnGroup(const qglviewer::Vec &pos, unsigned int nSeeweeds, subdivisionFunc getSubdivisions, randPosFunc generatePos) {

	unsigned int lastParticleId = this->getParticleWaitingCount();
	unsigned int nSubdivisions = _maxSubdivisions;
	unsigned int epsilon = 1;
	
	for (unsigned int i = 0; i < nSeeweeds; i++) {
		qglviewer::Vec position = Vec(pos.x + 10*Random::randf(), pos.y, pos.z + 10*Random::randf());

		for (unsigned int j = 0; j < nSubdivisions+1; j++) {
			qglviewer::Vec vel = Vec(Random::randf(-0.5,0.5), 0, Random::randf(-0.5,0.5));
			this->addParticle(new Particule(position + qglviewer::Vec(Random::randf(-0.3,0.3),j*epsilon,Random::randf(-0.3,0.3)), vel, 0.005, 0.001, j==0));	

			if(j>=1) {
				this->addSpring(lastParticleId + j - 1, lastParticleId + j, Random::randf(2,3),0.1,0.03,0.1);
			}
		}
			
		lastParticleId+= nSubdivisions+1;
	}
}


void SeeweedGroup::drawDownwards(const float *modelMatrix) {
	_seeweedsProgram->use();

	glUniformMatrix4fv(_seeweedsUniformLocs["modelMatrix"], 1, GL_TRUE, modelMatrix);
        
	glBindBufferBase(GL_UNIFORM_BUFFER, 0 , Globals::projectionViewUniformBlock);
	
	glBindBuffer(GL_ARRAY_BUFFER, springs_lines_b);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(0, 0);
	glEnableVertexAttribArray(0);
	
	glBindBuffer(GL_ARRAY_BUFFER, springs_intensity_b);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(1, 0);
	glEnableVertexAttribArray(1);
	
	glLineWidth(_seeweedWidth);
	glDrawArrays(GL_LINES, 0,nSprings*2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
}
		
void SeeweedGroup::makeSeeweedsProgram() {
	
	_seeweedsProgram = new Program("Seeweeds");
	
	_seeweedsProgram->bindAttribLocations("0 1", "pos intensity");
	_seeweedsProgram->bindFragDataLocation(0, "out_colour");
	_seeweedsProgram->bindUniformBufferLocations("0","projectionView");

	_seeweedsProgram->attachShader(Shader("shaders/seeweeds/vs.glsl", GL_VERTEX_SHADER));
	_seeweedsProgram->attachShader(Shader("shaders/seeweeds/fs.glsl", GL_FRAGMENT_SHADER));
	
	_seeweedsProgram->link();
	_seeweedsUniformLocs = _seeweedsProgram->getUniformLocationsMap("modelMatrix", true);
}
