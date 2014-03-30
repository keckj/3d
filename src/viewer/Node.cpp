#include "Node.h"

#include <iostream>
#include <string>
#include <vector>

Node::Node (std::string const& name) : name(name), vertices(NULL), normals(NULL), textures(NULL), nv(0), nn(0), nt(0) {
}

void Node::addSon (Node& son) {
    sons.push_back(son);
}

void Node::draw () {
    // First, draw the current node
    glPushMatrix();

    if (nn > 0) {
        glEnableClientState(GL_NORMAL_ARRAY);
    }
    glEnableClientState(GL_VERTEX_ARRAY);

    if (nn > 0) {
        glNormalPointer(GL_FLOAT, 0, normals);
    }
    glVertexPointer(3, GL_FLOAT, 0, vertices);

    glDrawArrays(GL_TRIANGLES, 0, nv / 3);

    glDisableClientState(GL_VERTEX_ARRAY);
    if (nn > 0) {
        glDisableClientState(GL_NORMAL_ARRAY);
    }

    glPopMatrix();

    // Then, its sons
    for (unsigned int i = 0; i < sons.size(); i++) {
        sons[i].draw();
    }
}

void Node::print () const {
    std::cout << "size vertices = " << nv << std::endl;
    std::cout << "size normals = " << nn << std::endl;

    std::cout << "vertices" << std::endl;
    for (unsigned int i = 0; i < nv; i++) {
        std::cout << vertices[i] << " ";
        if ((i + 1) % 3 == 0) {
            std::cout << std::endl;
        }
    }

    std::cout << "normals" << std::endl;
    for (unsigned int i = 0; i < nn; i++) {
        std::cout << normals[i] << " ";
        if ((i + 1) % 3 == 0) {
            std::cout << std::endl;
        }
    }
}
void Node::setVertices (GLfloat* vertices) {
    this->vertices = new GLfloat[nv];

    for (unsigned int i = 0; i < nv; i++) {
        this->vertices[i] = vertices[i];
    }
}

void Node::setNormals (GLfloat* normals) {
    this->normals = new GLfloat[nn];

    for (unsigned int i = 0; i < nn; i++) {
        this->normals[i] = normals[i];
    }
}

void Node::setTextures (GLfloat* textures) {
    this->textures = new GLfloat[nt];

    for (unsigned int i = 0; i < nt; i++) {
        this->textures[i] = textures[i];
    }
}

void Node::setName (std::string const& name) {
    this->name = name;
}

void Node::setSizeVertices (unsigned int nv) {
    this->nv = nv;
}

void Node::setSizeNormals (unsigned int nn) {
    this->nn = nn;
}

void Node::setSizeTextures (unsigned int nt) {
    this->nt = nt;
}

Node::~Node() {
    delete [] vertices;
    delete [] normals;
}

