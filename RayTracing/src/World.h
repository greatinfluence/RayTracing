#pragma once

#include "Camera.h"
#include "Cuboid.h"

class World {
public:
	World() : m_Cam(nullptr), m_Background(la::vec3(0)), m_Root{0} {}
	World(std::shared_ptr<Camera> cam, la::vec3 back = la::vec3(0.0f)):
		m_Cam(std::move(cam)), m_Background(back), m_Root{0} {}

	// AddGeo(geo) adds the geometry into the world
	void AddGeo(std::shared_ptr<Geometry> geo);
	
	// RayTracing(ray, lev, coef) traces the ray and returns the color seen by the ray
	__host__ __device__ la::vec3 RayTracing(Ray const& ray, int lev = 1, la::vec3 coef = la::vec3(1.0f));

	__host__ __device__ std::shared_ptr<Camera> GetCam() const { return m_Cam; }
	__host__ __device__ la::vec3 GetBackground() const { return m_Background; }

	// GetNgeo() returns the number of geometries in the world
	size_t GetNgeo() const { return m_Geos.size(); }

	std::shared_ptr<Geometry> const GetGeo(size_t ind) const { return m_Geos[ind].geo; }
	// Function to create the hierarchy of the geometries
	void CreateHierarchy();

	size_t GetRoot() const { return m_Root; }
private:
	struct Object{
		std::shared_ptr<Geometry> geo;
		la::vec3 cent;
		float area;

		Object(std::shared_ptr<Geometry> geo)
			: geo(geo), cent(la::vec3(0)), area(0) {
			ComputeInfo(geo.get(), cent, area);
		}
	};
	static void ComputeInfo(Geometry* geo, la::vec3& cent, float& area);
	size_t DoCreateHierarchy(size_t beg, size_t ed);
	std::vector<Object> m_Geos;
	size_t m_Root;
	std::shared_ptr<Camera> m_Cam;
	la::vec3 m_Background;
};
