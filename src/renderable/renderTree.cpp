
#include "renderTree.h"
#include "log.h"
#include "matrix.h"

#include <cassert>
#include <string>

RenderTree::RenderTree(bool active) 
: active(active)
{
	relativeModelMatrix = new float[16];
	setRelativeModelMatrix(consts::identity4);
}

RenderTree::~RenderTree() {
	delete [] relativeModelMatrix;
}
		
const float *RenderTree::getRelativeModelMatrix() const {
	return relativeModelMatrix;
}

void RenderTree::setRelativeModelMatrix(const float *matrix) {
	memcpy(relativeModelMatrix, matrix, 16*sizeof(float));
}


void RenderTree::addChild(std::string key, RenderTree *child) {
	assert(children.insert(std::pair<std::string,RenderTree*>(key, child)).second == true);	
}

void RenderTree::removeChild(std::string key) {
	assert(children.erase(key) == 1);
}


void RenderTree::desactivateChild(std::string childName) {
	std::map<std::string, RenderTree*>::iterator it;

	it = children.find(childName);
	assert(it != children.end());

	it->second->active = false;
}

void RenderTree::activateChild(std::string childName) {
	std::map<std::string, RenderTree*>::iterator it;

	it = children.find(childName);
	assert(it != children.end());

	it->second->active = false;
}



// RENDERABLE WRAPPER //
void RenderTree::draw() {
	draw(consts::identity4);
}

void RenderTree::draw(const float *currentTransformationMatrix) {
	
	//if not active draw nothing
	if(!this->active)
		return;
	
	//draw current object
	const float *newTransformationMatrix = Matrix::multMat4f(currentTransformationMatrix, this->getRelativeModelMatrix());

	this->drawDownwards(newTransformationMatrix);

	//draw subtrees
	std::map<std::string, RenderTree*>::const_iterator it;
	for (it = children.cbegin(); it != children.cend(); it++) {
		it->second->draw(newTransformationMatrix);
	}

	this->drawUpwards(newTransformationMatrix);

	delete [] newTransformationMatrix;
}

void RenderTree::animate() {
	if(!this->active)
		return;
	
	this->animateDownwards();

	//animate subtrees
	std::map<std::string, RenderTree*>::const_iterator it;
	for (it = children.cbegin(); it != children.cend(); it++) {
		it->second->animate();
	}

	this->animateUpwards();
}
		
void RenderTree::keyPressEvent(QKeyEvent* event, Viewer& v) {
	if(!this->active)
		return;
	
	this->keyPressEvent(event);

	//animate subtrees
	std::map<std::string, RenderTree*>::const_iterator it;
	for (it = children.cbegin(); it != children.cend(); it++) {
		it->second->keyPressEvent(event);
	}
}

void RenderTree::mouseMoveEvent(QMouseEvent* event, Viewer& v) { 
	if(!this->active)
		return;
	
	this->mouseMoveEvent(event);

	//animate subtrees
	std::map<std::string, RenderTree*>::const_iterator it;
	for (it = children.cbegin(); it != children.cend(); it++) {
		it->second->mouseMoveEvent(event);
	}
}
///////////////////////////

void RenderTree::translate(float x, float y, float z) {
	Matrix::translateMat4f(relativeModelMatrix,x,y,z);
}
void RenderTree::translate(qglviewer::Vec v) {
	Matrix::translateMat4f(relativeModelMatrix,v);
}
void RenderTree::scale(float alpha) {
	Matrix::scaleMat4f(relativeModelMatrix,alpha);
}
void RenderTree::scale(float alpha, float beta, float gamma) {
	Matrix::scaleMat4f(relativeModelMatrix,alpha,beta,gamma);
}
void RenderTree::scale(qglviewer::Vec v) {
	Matrix::scaleMat4f(relativeModelMatrix,v);
}
void RenderTree::rotate(qglviewer::Quaternion rot) {
	Matrix::rotateMat4f(relativeModelMatrix,rot);
}
void RenderTree::pushMatrix(const float *matrix) {
	float *tmp = Matrix::multMat4f(matrix, relativeModelMatrix);	
	delete [] relativeModelMatrix;
	relativeModelMatrix = tmp;
}

void RenderTree::translateChild(std::string childName, float x, float y, float z) {
	Matrix::translateMat4f(children[childName]->relativeModelMatrix,x,y,z);
}
void RenderTree::translateChild(std::string childName, qglviewer::Vec v) {
	Matrix::translateMat4f(children[childName]->relativeModelMatrix,v);
}
void RenderTree::scaleChild(std::string childName, float alpha) {
	Matrix::scaleMat4f(children[childName]->relativeModelMatrix,alpha);
}
void RenderTree::scaleChild(std::string childName, float alpha, float beta, float gamma) {
	Matrix::scaleMat4f(children[childName]->relativeModelMatrix,alpha,beta,gamma);
}
void RenderTree::scaleChild(std::string childName, qglviewer::Vec v) {
	Matrix::scaleMat4f(children[childName]->relativeModelMatrix,v);
}
void RenderTree::rotateChild(std::string childName, qglviewer::Quaternion rot) {
	Matrix::rotateMat4f(children[childName]->relativeModelMatrix,rot);
}
void RenderTree::pushMatrixToChild(std::string childName, const float *matrix) {
	float *src = children[childName]->relativeModelMatrix;
	float *tmp = Matrix::multMat4f(matrix, src);	
	delete [] src;

	children[childName]->relativeModelMatrix = tmp;
}
