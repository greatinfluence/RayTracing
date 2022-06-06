#pragma once

#include "Ball.h"
#include "Triangle.h"
#include "Camera.h"

class World {
public:
	World() = delete;
	World(Camera cam, glm::vec3 back = glm::vec3(0.0f)): m_Cam(cam), m_Background(back) {}
	void AddGeo(std::shared_ptr<Geometry> geo);
	glm::vec3 RayTracing(Ray ray, int lev = 0, glm::vec3 coef = glm::vec3(1.0f));
	Camera& Getcam() { return m_Cam; }
private:
	std::vector<std::shared_ptr<Geometry>> m_Geos;
	Camera m_Cam;
	glm::vec3 m_Background;
};
