#include <cmath>
#include <iostream>
#include <map>

using namespace std;

#include "viewer.h"
#include "dynamicSystem.h"

DynamicSystem::DynamicSystem()
{
    // global scene parameters
    numberParticles = 10;
    defaultGravity = Vec(0.0, 0.0, -10.0);
    gravity = defaultGravity;
    defaultMediumViscosity = 1.0;
    mediumViscosity = defaultMediumViscosity;
    handleCollisions = true;
    dt = 0.1;
    groundPosition = Vec(0.0, 0.0, 0.0);
    groundNormal = Vec(0.0, 0.0, 1.0);
    rebound = 0.5;
    // parameters shared by all particles
    particleMass = 1.0;
    particleRadius = 0.25;
    distanceBetweenParticles = 4.0 * particleRadius;
    // parameters shared by all springs
    springStiffness = 30.0;
    springInitLength = 0.5;
    springDamping = 1.0;
    // toggles
    toggleGravity = true;
    toggleViscosity = true;
    toggleCollisions = true;
}

DynamicSystem::~DynamicSystem()
{
    clear();
}


void DynamicSystem::clear()
{
    vector<Particle *>::iterator itF;
    for (itF = fixed.begin(); itF != fixed.end(); ++itF) {
        delete(*itF);
    }
    fixed.clear();

    vector<Particle *>::iterator itP;
    for (itP = particles.begin(); itP != particles.end(); ++itP) {
        delete(*itP);
    }
    particles.clear();

    vector<Spring *>::iterator itS;
    for (itS = springs.begin(); itS != springs.end(); ++itS) {
        delete(*itS);
    }
    springs.clear();
}

const Vec &DynamicSystem::getFixedParticlePosition() const
{
    return fixed[0]->getPosition();	// no check on 0!
}

void DynamicSystem::setFixedParticlePosition(const Vec &pos)
{
    if (fixed.size() > 0)
        fixed[0]->setPosition(pos);
}

void DynamicSystem::setGravity(bool onOff)
{
    gravity = (onOff ? defaultGravity : Vec());
}

void DynamicSystem::setViscosity(bool onOff)
{
    mediumViscosity = (onOff ? defaultMediumViscosity : 0.0);
}

void DynamicSystem::setCollisionsDetection(bool onOff)
{
    handleCollisions = onOff;
}


void DynamicSystem::init(Viewer &viewer)
{
    clear();

    // add a manipulatedFrame to move particle 0 with the mouse
    viewer.setManipulatedFrame(new qglviewer::ManipulatedFrame());
    viewer.manipulatedFrame()->setPosition(getFixedParticlePosition());
}

void DynamicSystem::draw()
{
    // Fixed Particles
    vector<Particle *>::iterator itP;
    for (itP = fixed.begin(); itP != fixed.end(); ++itP) {
        glColor3fv((*itP)->getColor());
        (*itP)->draw();
    }

    // Particles
    for (itP = particles.begin(); itP != particles.end(); ++itP) {
        glColor3fv((*itP)->getColor());
        (*itP)->draw();
    }

    // Springs
    glColor3f(1.0, 0.28, 0.0);
    glLineWidth(5.0);
    vector<Spring *>::iterator itS;
    for (itS = springs.begin(); itS != springs.end(); ++itS) {
        (*itS)->draw();
    }

    // Ground
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_QUADS);
    glVertex3f(-10.0f, -10.0f, groundPosition.z);
    glVertex3f(-10.0f, 10.0f,  groundPosition.z);
    glVertex3f( 10.0f, 10.0f,  groundPosition.z);
    glVertex3f( 10.0f, -10.0f, groundPosition.z);
    glEnd();

    glColor3f(1.0, 1.0, 1.0);
}


void DynamicSystem::animate()
{

    //======== 1. Compute all forces
    // map to accumulate the forces to apply on each particle
    map<const Particle *, Vec> forces;

    // weights
    vector<Particle *>::iterator itP;
    for (itP = particles.begin(); itP != particles.end(); ++itP) {
        Particle *p = *itP;
        forces[p] = gravity * p->getMass();
    }

    // viscosity
    for (itP = particles.begin(); itP != particles.end(); ++itP) {
        Particle *p = *itP;
        forces[p] += -mediumViscosity * p->getVelocity();
    }

    // damped springs
    vector<Spring *>::iterator itS;
    for (itS = springs.begin(); itS != springs.end(); ++itS) {
        Spring *s = *itS;
        Vec f12 = s->getCurrentForce();
        forces[s->getParticle1()] += f12;
        forces[s->getParticle2()] -= f12; // opposite force
    }


    //======== 2. Integration scheme
    // update particles velocity (qu. 1)
    for (itP = particles.begin(); itP != particles.end(); ++itP) {
        Particle *p = *itP;
        Vec force = forces[p];
        // v = v + h * f / m
        p->incrVelocity(dt * force * p->getInvMass());
    }

    // update particles positions
    for (itP = particles.begin(); itP != particles.end(); ++itP) {
        Particle *p = *itP;
        // q = q + dt * v
        p->incrPosition(dt * p->getVelocity());
    }


    //======== 3. Collisions
    if (handleCollisions) {
        //TO DO: discuss multi-collisions and order!
        for (itP = particles.begin(); itP != particles.end(); ++itP) {
            collisionParticleGround(*itP);
        }
        for(unsigned int i = 1; i < particles.size(); ++i) {
            Particle *p1 = particles[i - 1];
            Particle *p2 = particles[i];
            collisionParticleParticle(p1, p2);
        }
    }
}

void DynamicSystem::collisionParticleGround(Particle *p)
{
    // don't process fixed particles (ground plane is fixed)
    if (p->getInvMass() == 0)
        return;

    // particle-plane distance
    double penetration = (p->getPosition() - groundPosition) * groundNormal;
    penetration -= p->getRadius();
    if (penetration >= 0)
        return;

    // penetration velocity
    double vPen = p->getVelocity() * groundNormal;

    // updates position and velocity of the particle
    p->incrPosition(-penetration * groundNormal);
    p->incrVelocity(-(1 + rebound) * vPen * groundNormal);
}


void DynamicSystem::collisionParticleParticle(Particle *p1, Particle *p2)
{
    // TODO!
}

void DynamicSystem::keyPressEvent(QKeyEvent* e, Viewer& viewer)
{
    // Get event modifiers key
    const Qt::KeyboardModifiers modifiers = e->modifiers();

    /* Controls added for Lab Session 6 "Physicall Modeling" */
    if ((e->key()==Qt::Key_G) && (modifiers==Qt::NoButton)) {
        toggleGravity = !toggleGravity;
        setGravity(toggleGravity);
        viewer.displayMessage("Set gravity to "
                              + (toggleGravity ? QString("true") : QString("false")));

    } else if ((e->key()==Qt::Key_V) && (modifiers==Qt::NoButton)) {
        toggleViscosity = !toggleViscosity;
        setViscosity(toggleViscosity);
        viewer.displayMessage("Set viscosity to "
                              + (toggleViscosity ? QString("true") : QString("false")));

    } else if ((e->key()==Qt::Key_C) && (modifiers==Qt::NoButton)) {
        toggleCollisions = !toggleCollisions;
        setCollisionsDetection(toggleCollisions);
        viewer.displayMessage("Detects collisions "
                              + (toggleCollisions ? QString("true") : QString("false")));

    } else if ((e->key()==Qt::Key_Home) && (modifiers==Qt::NoButton)) {
        // stop the animation, and reinit the scene
        viewer.stopAnimation();
        init(viewer);
        viewer.manipulatedFrame()->setPosition(getFixedParticlePosition());
        toggleGravity = true;
        toggleViscosity = true;
        toggleCollisions = true;
    }
}

void DynamicSystem::mouseMoveEvent(QMouseEvent*, Viewer& v)
{
    /* setFixedParticlePosition(v.manipulatedFrame()->position()); */
}

