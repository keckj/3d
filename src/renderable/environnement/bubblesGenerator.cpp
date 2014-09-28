
#include "headers.h"
#include "bubblesGenerator.h"
#include "constantForce.h"
#include "dynamicScheme.h"
#include "killParticles.h"
#include "seaFlow.h"
#include "particule.h"
#include "rand.h"
#include <sstream>

BubblesGenerator::BubblesGenerator(
		unsigned int nBubbles, unsigned int nGroups, 
		unsigned int generationFrequency, unsigned int memoryFactor) :
		_nBubbles(nBubbles), _nGroups(nGroups), 
		_generationFrequency(generationFrequency),
		_memoryFactor(memoryFactor) 
	{
		_groups = new ParticleGroup*[_nGroups];
        
        qglviewer::Vec g = 0.0001*Vec(0,+9.81,0);
		ParticleGroupKernel *archimede = new ConstantForce(g);
        ParticleGroupKernel *dynamicScheme = new DynamicScheme();
        ParticleGroupKernel *killBubbles = new KillParticles(qglviewer::Vec(0,1,0), 10);
		ParticleGroupKernel *seaflow = new SeaFlow(qglviewer::Vec(1,0,0), 0.002, 0.001);

		std::stringstream name;
        for (unsigned int i = 0; i < _nGroups; i++) {
                _groups[i] = new ParticleGroup(_nBubbles*memoryFactor, 1);
                _groups[i]->addKernel(archimede);
                _groups[i]->addKernel(seaflow);
                _groups[i]->addKernel(killBubbles);
                _groups[i]->addKernel(dynamicScheme);
                name.clear();
                name << "particles";
                name << i;
                this->addChild(name.str(), _groups[i]);
        }
	}

BubblesGenerator::~BubblesGenerator() {
	for (unsigned int i = 0; i < _nGroups; i++) {
		delete _groups[i];
	}

	delete [] _groups;
}

void BubblesGenerator::drawDownwards(const float *currentTransformationMatrix) {
}

void BubblesGenerator::animateDownwards() {
	static unsigned int frameCount = 0;

	if(frameCount % _generationFrequency == 0)
		generateBubbles();

	frameCount++;
}

void BubblesGenerator::generateBubbles() {
        
	for (unsigned int j = 0; j < _nGroups; j++) {
                ParticleGroup *p = _groups[j];
                qglviewer::Vec pos = Vec(Random::randf(-45,40), -30, Random::randf(0,25));
                for (unsigned int i = 0; i < _nBubbles; i++) {
                                qglviewer::Vec  vel = Vec(0,0,0);
                                float r = Random::randf(0.04,0.15);
                                float m = 4.0f/3.0f*3.1415f*r*r*r;
                                p->addParticle(new Particule(pos + qglviewer::Vec(Random::randf(), Random::randf(), Random::randf()), vel, m, r, false));	
                }
                
                p->releaseParticles();
        }
}
