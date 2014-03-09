#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <vector>

#include "ObjLoader.h"
#include "Vector3f.h"

ObjLoader::ObjLoader (std::string const& filename) : filename(filename), node(filename) {
    parse();
    print();
}

void ObjLoader::print () {
    node.print();
}

void ObjLoader::parse () {
    std::ifstream file(filename.c_str());

    std::vector<int> iv, in, it;
    std::vector<Vector3f> ver, nor, tex;
    std::vector<std::string> face;

    // vertex : v x y z
    std::string line;
    while (getline(file, line)) {
        // skip comments and blank lines
        if (line[0] == '#' || line == "")
            continue;

        switch (line[0]) {
            // vertex, normal or texture
            case 'v':
                switch (line[1]) {
                    case ' ':
                        // vertex
                        float x, y, z;
                        strToVector(line.substr(2), x, y, z);

                        ver.push_back(Vector3f(x, y, z));
                        break;

                    case 'n':
                        // normal
                        float xn, yn, zn;
                        strToVector(line.substr(2), xn, yn, zn);

                        nor.push_back(Vector3f(xn, yn, zn));
                        break;

                    case 't':
                        // textures
                        float xt, yt, zt;
                        strToVector(line.substr(2), xt, yt, zt);

                        tex.push_back(Vector3f(xt, yt));
                        break;

                    default:
                        break;
                }
                break;

            // face
            case 'f':
                face.clear();
                face = splitOnWS(line.substr(2));
                int v, vt, vn;

                for (unsigned int i = 0; i < face.size(); i++) {
                    switch (countSlashes(face[i])) {
                        case 0:
                            sscanf(face[i].c_str(), "%d", &v);
                            vt = 0;
                            vn = 0;
                            break;

                        case 2:
                            sscanf(face[i].c_str(), "%d/%d/%d", &v, &vt, &vn);
                            break;

                        default:
                            std::cerr << "Unsuported format" << std::endl;
                            break;
                    }
                    iv.push_back(v - 1);
                    it.push_back(vt - 1);
                    in.push_back(vn - 1);
                }
                break;

            default:
                break;
        }
    }

    file.close();
    // Parsing is done!
    // Now, we have to create vectors of vertices, normals and textures in the correct order
    std::vector<GLfloat> tv, tn, tt;
    for (unsigned int i = 0; i < iv.size(); i++) {
        if (iv[i] < int(ver.size())) {
            tv.push_back(ver[iv[i]].getX());
            tv.push_back(ver[iv[i]].getY());
            tv.push_back(ver[iv[i]].getZ());
        }
    }

    for (unsigned int i = 0; i < in.size(); i++) {
        if (in[i] < int(nor.size()) && in[i] != -1) {
            tn.push_back(nor[in[i]].getX());
            tn.push_back(nor[in[i]].getY());
            tn.push_back(nor[in[i]].getZ());
        }
    }

    for (unsigned int i = 0; i < it.size(); i++) {
        if (it[i] < int(tex.size()) && it[i] != -1) {
            tt.push_back(tex[it[i]].getX());
            tt.push_back(tex[it[i]].getZ());
        }
    }

    // Everything is in the appropriate order, we convert the vectors into GLfloat*
    node.setSizeVertices(tv.size());
    node.setSizeNormals(tn.size());
    node.setSizeTextures(tt.size());

    node.setVertices(vector2float(tv));
    node.setNormals(vector2float(tn));
    node.setTextures(vector2float(tt));

    ver.clear();
    nor.clear();
    tex.clear();

    iv.clear();
    in.clear();
    it.clear();
}

/* Convert string to float */
void ObjLoader::strToVector (std::string const& str, float &x, float &y, float &z) {
    std::stringstream ss(str);
    ss >> x >> y >> z;
}

/* Count the number of '/' in a string */
int ObjLoader::countSlashes (std::string const& str) {
    int res = 0;

    for (unsigned int i = 0; i < str.size(); i++) {
        if (str[i] == '/') {
            res++;
        }
    }

    return res;
}

/* Convert a vector of a float into a GLfloat array */
GLfloat* ObjLoader::vector2float (std::vector<float>& array) {
    GLfloat *res = new GLfloat[array.size()];

    for (unsigned int i = 0; i < array.size(); i++) {
        res[i] = array[i];
    }

    return res;
}

/* Split a string on whitespaces */
std::vector<std::string> ObjLoader::splitOnWS (std::string const& str) {
    std::vector<std::string> res;
    std::string tmp;

    for (unsigned int i = 0; i < str.size(); i++) {
        if (str[i] == ' ') {
            if (tmp != "") {
                res.push_back(tmp);
                tmp = "";
            }
        } else {
            tmp += str[i];
        }
    }
    res.push_back(tmp);

    return res;
}

void ObjLoader::draw () {
    node.draw();
}

