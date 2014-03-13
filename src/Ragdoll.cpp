#include "Ragdoll.h"

Ragdoll::Ragdoll () {
}

void Ragdoll::addPart (std::string const& name, BodyPart* part) {
    parts[name] = part;
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

