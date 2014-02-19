#ifndef VECTOR3F_H
#define VECTOR3F_H

class Vector3f {
    public:
        Vector3f (float x = 0, float y = 0, float z = 0);
        float getX () const;
        float getY () const;
        float getZ () const;

    private:
        float x;
        float y;
        float z;
};

#endif
