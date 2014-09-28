
#include "particleGroup.h"
#include "renderTree.h"

class BubblesGenerator : public RenderTree {
	
	public:
		BubblesGenerator(unsigned int nBubbles, unsigned int nGroups, unsigned int generationFrequency, unsigned int memoryFactor);
		~BubblesGenerator();

		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
		void animateDownwards();

	private:
		unsigned int _nBubbles, _nGroups, _generationFrequency, _memoryFactor;
        ParticleGroup **_groups;

		void generateBubbles();
};
