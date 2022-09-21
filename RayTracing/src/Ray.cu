#include "Ray.h"

//#include <cassert>

#include "la.h"

__device__ inline static float Hitball(Ray const& ray, Ball const* ball) {
	la::vec3 const& m_Pos = ray.m_Pos;
	la::vec3 const& m_Dir = ray.m_Dir;
	float r = ball->m_Radius;
	la::vec3 const& cent = ball->m_Center;
	la::vec3 const diff = cent - m_Pos;
	float d = la::l2Norm(diff);
	if (d < r) {
		// Inside the ball
		if (d < eps) {
			// At the center
			return r;
		}
		float cosine = la::dot(la::normalize(diff), m_Dir);
		return d * cosine + sqrtf(sq(d * cosine) + sq(r) - sq(d));
	}
	else {
		// Outside the ball
		if (la::dot(diff, m_Dir) < 0) {
			// leaving the ball
			return floatmax;
		}
		la::vec3 pdiff = la::perp(diff, m_Dir);
		if (la::l2Norm(pdiff) > r) {
			// Out of range
			return floatmax;
		}
		la::vec3 tolen = la::proj(diff, m_Dir);
		return la::l2Norm(tolen) - sqrtf(sq(r) - la::dot(pdiff, pdiff));
	}
}

__device__ inline static float Hitcuboid(Ray const& ray, Cuboid const* cub) {
//	printf("In Cuboid\n");
	/*
		This method is modified from http://tog.acm.org/resources/GraphicsGems/gems/RayBox.c
	*/
	la::vec3 const& m_Pos = ray.m_Pos;
	la::vec3 const& m_Dir = ray.m_Dir;
	bool inside = true; // Denote if the ray is inside the box

	// quadrant denote if the starting position is inside the box
	enum class Quadrant {
		Left = 0,
		Middle = 1,
		Right = 2,
	} quadrant[3] = { Quadrant::Middle, Quadrant::Middle, Quadrant::Middle };
	float candidatePlane[3] = {};
	int whichPlane = 0;
	float diff[3] = {};

	for (auto i = 0; i < 3; ++i) {
		float min = cub->GetMin(i), max = cub->GetMax(i);
		if (m_Pos[i] < min) {
			quadrant[i] = Quadrant::Left;
			candidatePlane[i] = min;
			inside = false;
		}
		else if (m_Pos[i] > max) {
			quadrant[i] = Quadrant::Right;
			candidatePlane[i] = max;
			inside = false;
		}
	}

	if (inside) {
		// Ray is inside the box
//		printf("Inside the box\n");
		return 0.0f;
	}

	// Calculate the dist to the plane
	for (auto i = 0; i < 3; ++i) {
		if (quadrant[i] != Quadrant::Middle && fabs(m_Dir[i]) > 1e-6) {
			diff[i] = (candidatePlane[i] - m_Pos[i]) / m_Dir[i];
		}
		else diff[i] = -1.0f;
	}
	
	// Get largest of the dist for final choice of intersection
	for (auto i = 0; i < 3; ++i) {
		if (diff[whichPlane] < diff[i]) whichPlane = i;
	}

//	printf("Last check\n");

	// Check if the ray really hit the box
	if (diff[whichPlane] < 0.0f) return floatmax;
	for (auto i = 0; i < 3; ++i) {
		if (whichPlane != i) {
			float hitpos = m_Pos[i] + diff[whichPlane] * m_Dir[i];
			if (hitpos < cub->GetMin(i) || hitpos > cub->GetMax(i))
				return floatmax;
		}
	}
//	printf("Touches the Cuboid\n");
	return diff[whichPlane];
}

__device__ static float Hittriangle(Ray const& ray, Triangle const* triangle) {

	auto const& m_Pos = ray.m_Pos;
	auto const& m_Dir = ray.m_Dir;
//	assert(fabs(m_Dir.x) < 1.5f);
	//assert(fabs(m_Dir.y) < 1.5f);
	//assert(fabs(m_Dir.z) < 1.5f);
	la::vec3 norm = triangle->GetNorm(m_Pos);
	la::vec3 pp = proj(triangle->GetPos(0) - m_Pos, norm);
	if (l2Norm(pp) < eps) {
		// Too close to the triangle
		return floatmax;
	}
	float cosval = la::dot(pp, m_Dir) / l2Norm(pp);
	if (cosval < eps) {
		// Leaving or perpendicular to the plane of the triangle
		return floatmax;
	}
	float dist = l2Norm(pp) / cosval;
	la::vec3 pos = m_Pos + m_Dir * dist;
	la::vec3 vec = pos - triangle->GetPos(0);
	float val = l2Norm(proj(vec, norm));
//	assert(val < 1e-4);
	if (triangle->OnTriangle(m_Pos + m_Dir * dist)) {
		return dist;
	}
	else return floatmax;
}

__device__ inline static float Hitdevice(Ray const& ray, Geometry const* geo) {
///	la::vec3 m_Pos = ray.GetPos();
//	la::vec3 m_Dir = ray.GetDir();
//	printf("%f\n", m_Pos.x);
	if (geo == nullptr) {
		printf("Ray::Hit error: Received nullptr\n");
		return -1;
	}
//	printf("Enter Ray::Hit\n");
//	printf("0x%p\n", geo);
	//printf("(%f %f %f) -> (%f %f %f)\n", ray.GetPos().x, ray.GetPos().y, ray.GetPos().z, ray.GetDir().x, ray.GetDir().y, ray.GetDir().z);
	switch (geo->GetType()) {
	case GeoType::Ball: {
		return Hitball(ray, static_cast<Ball const*>(geo));
	}
	case GeoType::Triangle: {
		return Hittriangle(ray, static_cast<Triangle const*>(geo));
	}
	case GeoType::Cuboid: {
		return Hitcuboid(ray, static_cast<Cuboid const*>(geo));
	}
	default: {
		printf("Not implemented yet!\n");
	}
	}
	return 0.0f;
}

float Hithost(Ray const& ray, Geometry const* geo) {
	auto const& m_Pos = ray.m_Pos;
	auto const& m_Dir = ray.m_Dir;
	switch (geo->GetType()) {
	case GeoType::Ball: {
		auto* ball = static_cast<Ball const*> (geo);
		float r = ball->m_Radius;
		la::vec3 cent = ball->m_Center;
		float d = la::l2Norm(cent - m_Pos);
		if (d < r) {
			// Inside the ball
			if (d < eps) {
				// At the center
				return r;
			}
			float cosine = la::dot(la::normalize(cent - m_Pos), m_Dir);
			return (float)(d * cosine + sqrt(sq(d * cosine) + sq(r) - sq(d)));
		}
		else {
			// Outside the ball
			if (la::dot(cent - m_Pos, m_Dir) < 0) {
				// leaving the ball
				return std::numeric_limits<float>::max();
			}
			la::vec3 diff = la::perp(cent - m_Pos, m_Dir);
			if (la::l2Norm(diff) > r) {
				// Out of range
				return std::numeric_limits<float>::max();
			}
			la::vec3 tolen = la::proj(cent - m_Pos, m_Dir);
			return la::l2Norm(tolen) - sqrt(sq(r) - la::dot(diff, diff));
		}
	}
	case GeoType::Triangle: {
		auto* triangle = static_cast<Triangle const*>(geo);
		la::vec3 norm = triangle->GetNorm(m_Pos);
		la::vec3 pp = la::proj(triangle->GetPos(0) - m_Pos, norm);
		if (la::l2Norm(pp) < eps) {
			// Too close to the triangle
			return std::numeric_limits<float>::max();
		}
		float cosval = la::dot(pp, m_Dir) / la::l2Norm(pp);
		if (cosval < eps) {
			// Leaving or perpendicular to the plane of the triangle
			return std::numeric_limits<float>::max();
		}
		float dist = la::l2Norm(pp) / cosval;
		la::vec3 pos = m_Pos + m_Dir * dist;
		la::vec3 vec = pos - triangle->GetPos(0);
		float val = la::l2Norm(la::proj(vec, norm));
		if (triangle->OnTriangle(m_Pos + m_Dir * dist)) {
			return dist;
		}
		else return std::numeric_limits<float>::max();
	}
	case GeoType::Cuboid: {
		/*
			This method is modified from http://tog.acm.org/resources/GraphicsGems/gems/RayBox.c
		*/
		auto* cub = static_cast<Cuboid const*>(geo);
		bool inside = true; // Denote if the ray is inside the box

		// quadrant denote if the starting position is inside the box
		enum class Quadrant {
			Left = 0,
			Middle = 1,
			Right = 2,
		} quadrant[3] = { Quadrant::Middle, Quadrant::Middle, Quadrant::Middle };
		float candidatePlane[3] = {};
		int whichPlane = 0;
		float diff[3] = {};

		for (auto i = 0; i < 3; ++i) {
			float min = cub->GetMin(i), max = cub->GetMax(i);
			if (m_Pos[i] < min) {
				quadrant[i] = Quadrant::Left;
				candidatePlane[i] = min;
				inside = false;
			}
			else if (m_Pos[i] > max) {
				quadrant[i] = Quadrant::Right;
				candidatePlane[i] = max;
				inside = false;
			}
		}

		if (inside) {
			// Ray is inside the box
			return 0.0f;
		}

		// Calculate the dist to the plane
		for (auto i = 0; i < 3; ++i) {
			if (quadrant[i] != Quadrant::Middle && fabs(m_Dir[i]) > 1e-6) {
				diff[i] = (candidatePlane[i] - m_Pos[i]) / m_Dir[i];
			}
			else diff[i] = -1.0f;
		}

		// Get largest of the dist for final choice of intersection
		for (auto i = 0; i < 3; ++i) {
			if (diff[whichPlane] < diff[i]) whichPlane = i;
		}

		// Check if the ray really hit the box
		if (diff[whichPlane] < 0.0f) return std::numeric_limits<float>().max();
		for (auto i = 0; i < 3; ++i) {
			if (whichPlane != i) {
				float hitpos = m_Pos[i] + diff[whichPlane] * m_Dir[i];
				if (hitpos < cub->GetMin(i) || hitpos > cub->GetMax(i))
					return std::numeric_limits<float>().max();
			}
		}
		return diff[whichPlane];
	}
	default: {
		printf("Not implemented yet!\n");
	}
}
	return 0.0f;
}

__host__ __device__ float Ray::Hit(Geometry const* geo) const
{
#ifdef __CUDA_ARCH__
	// device
//	printf("(%f %f %f) -> (%f %f %f)\n", m_Pos.x, m_Pos.y, m_Pos.z, m_Dir.x, m_Dir.y, m_Dir.z);
	return Hitdevice(*this, geo);
#else
	// host
	return Hithost(*this, geo);
#endif
}
