#include "World.h"

#include <limits>
#include <iostream>

#include "Random.h"
#include "Testtool.hpp"

void World::AddGeo(std::shared_ptr<Geometry> geo)
{
	m_Geos.push_back(geo);
}

glm::vec3 World::RayTracing(Ray ray, int lev, glm::vec3 coef)
{
	if(lev > 2) {
		float rr = fmax(coef.r, fmax(coef.g, coef.b));
		if (Random::Rand(1.0f) > rr) return coef * m_Background;
		coef = coef / rr;
	}
	float dist = std::numeric_limits<float>::max();
	std::shared_ptr<Geometry> hitted = nullptr;
	for (auto geo : m_Geos) {
		float ndis = ray.Hit(geo.get());
		if (ndis < dist) {
			dist = ndis;
			hitted = geo;
		}
	}
	if (hitted != nullptr) {
		glm::vec3 hitpos = ray.GetPos() + ray.GetDir() * dist;
		glm::vec3 att, wi, norm = hitted->GetNorm(hitpos);
		std::shared_ptr<Material> mat = hitted->GetMaterial();
		float poss = mat->scatter(hitpos, -ray.GetDir(),
			norm, att, wi);
		//std::cout << "Hittttttttttttttttttttttttt" << std::endl;
		//std::cout << coef * att * glm::dot(wi, norm) / poss << std::endl;
		return coef * mat->GetGlow() +
			RayTracing(Ray(hitpos + wi * 1e-3f, wi), lev + 1, coef * att * glm::dot(wi, norm) / poss);
	}
	else return coef * m_Background;
}
