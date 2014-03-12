#include "BodyPart.h"

BodyPart::BodyPart (bool enabled) : enabled(enabled) {
}

void BodyPart::disable () {
    enabled = false;
}

void BodyPart::enable () {
    enabled = false;
}

bool BodyPart::isEnabled () const {
    return enabled;
}

BodyPart::~BodyPart () {
}

