#include <iostream>
#include <map>

#include "Ragdoll.h"

Ragdoll::Ragdoll () {
}

void Ragdoll::addPart (std::string const& name, BodyPart* part) {
    parts[name] = part;
}

BodyPart* Ragdoll::getPart (std::string const& name) const {
    std::map<std::string, BodyPart*>::const_iterator it = parts.find(name);

    if (it == parts.end()) {
        std::cerr << name << " not found!" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        return it->second;
    }
}

void Ragdoll::disablePart (std::string const& name) {
    parts[name]->disable();
}

void Ragdoll::enablePart (std::string const& name) {
    parts[name]->enable();
}

void Ragdoll::removePart (std::string const& name) {
    parts.erase(name);
}

void Ragdoll::draw () {
    for (std::map<std::string, BodyPart*>::const_iterator it = parts.begin(); it != parts.end(); it++) {
        if (it->second->isEnabled()) {
            std::cout << "Drawing : " << it->first << std::endl;
            it->second->draw();
        }
    }
}

void Ragdoll::animate () {
    for (std::map<std::string, BodyPart*>::const_iterator it = parts.begin(); it != parts.end(); it++) {
        if (it->second->isEnabled()) {
            it->second->animate();
        }
    }
}

