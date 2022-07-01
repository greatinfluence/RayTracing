#pragma once

#include "Camera.h"
#include "Cuboid.h"

class World {
public:
	World() : m_Cam(), m_Background(glm::vec3(0)), m_Root{0} {}
	World(Camera cam, glm::vec3 back = glm::vec3(0.0f)) : m_Cam(cam), m_Background(back), m_Root{0} {}

	// AddGeo(geo) adds the geometry into the world
	void AddGeo(std::shared_ptr<Geometry> geo);
	
	// RayTracing(ray, lev, coef) traces the ray and returns the color seen by the ray
	glm::vec3 RayTracing(Ray ray, int lev = 1, glm::vec3 coef = glm::vec3(1.0f));

	Camera const& GetCam() const { return m_Cam; }
	glm::vec3 GetBackground() const { return m_Background; }

	// GetNgeo() returns the number of geometries in the world
	size_t GetNgeo() const { return m_Geos.size(); }

	std::shared_ptr<Geometry> const GetGeo(size_t ind) const { return m_Geos[ind].geo; }
	// Function to create the hierarchy of the geometries
	void CreateHierarchy();
private:
	struct Object{
		std::shared_ptr<Geometry> geo;
		glm::vec3 cent;
		float area;

		Object(std::shared_ptr<Geometry> geo)
			: geo(geo), cent(glm::vec3(0)), area(0) {
			ComputeInfo(geo.get(), cent, area);
		}
	};
	static void ComputeInfo(Geometry* geo, glm::vec3& cent, float& area);
	size_t DoCreateHierarchy(size_t beg, size_t ed);
	std::vector<Object> m_Geos;
	size_t m_Root;
	Camera m_Cam;
	glm::vec3 m_Background;
};
