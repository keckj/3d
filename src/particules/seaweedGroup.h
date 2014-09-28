
#include "headers.h"
#include "particleGroup.h"

typedef const qglviewer::Vec &(*randPosFunc)(const qglviewer::Vec &originalPos);
typedef unsigned int (*subdivisionFunc)(const qglviewer::Vec &pos);

class SeeweedGroup : public ParticleGroup {

	public:
		SeeweedGroup(unsigned int maxSeeweeds, unsigned int maxSubdivisions, float seeweedWidth);
		~SeeweedGroup();

		void spawnGroup(const qglviewer::Vec &pos, unsigned int nSeeweeds, subdivisionFunc getSubdivisions, randPosFunc generatePos);

		void drawDownwards(const float *modelMatrix = consts::identity4);

	private:
		Program *_seeweedsProgram;
		std::map<std::string, int> _seeweedsUniformLocs;

		unsigned int _maxSeeweeds, _maxSubdivisions;
		float _seeweedWidth;
		
		void makeSeeweedsProgram();
};
