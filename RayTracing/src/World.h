#pragma once

#include "Ball.h"
#include "Triangle.h"
#include "Camera.h"

class World {
public:
	World(): m_Cam(), m_Background(glm::vec3(0)) {}
	World(Camera cam, glm::vec3 back = glm::vec3(0.0f)): m_Cam(cam), m_Background(back) {}
	void AddGeo(std::shared_ptr<Geometry> geo);
	glm::vec3 RayTracing(Ray ray, int lev = 1, glm::vec3 coef = glm::vec3(1.0f));
	Camera const& GetCam() const { return m_Cam; }
	glm::vec3 GetBackground() const { return m_Background; }
	std::vector<std::shared_ptr<Geometry>> const GetGeos() const { return m_Geos; }
private:
	std::vector<std::shared_ptr<Geometry>> m_Geos;
	Camera m_Cam;
	glm::vec3 m_Background;
};
