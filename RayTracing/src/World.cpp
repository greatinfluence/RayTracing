#include "World.h"

#include <limits>
#include <algorithm>

#include "Random.h"
#include "Testtool.h"
#include "Triangle.h"
#include "Ball.h"
#include "Geometryrepository.h"

void World::AddGeo(std::shared_ptr<Geometry> geo)
{
	m_Geos.push_back(geo);
}

glm::vec3 World::RayTracing(Ray ray, int lev, glm::vec3 coef)
{
	float dist = std::numeric_limits<float>::max();
	Geometry* hitted = nullptr;
	if (fabs(ray.GetDir().x) > 1.5f) {
		printf("What?\n");
	}
	static_cast<Cuboid*>(Geometryrepository::GetGeo(m_Root))->TestHit(ray, dist, hitted);
	if (hitted != nullptr) {
		glm::vec3 hitpos = ray.GetPos() + ray.GetDir() * dist;
		glm::vec3 att, wi, norm = hitted->GetNorm(hitpos);
		Material* mat = hitted->GetMaterial();
		float poss = mat->scatter(hitpos, -ray.GetDir(),
			norm, att, wi);
		//std::cout << coef * att * glm::dot(wi, norm) / poss << std::endl;
		if (fabs(dist) > 1e3 || fabs(ray.GetDir().x) > 1.5f) {
			// Impossible!
			printf("What?\n");
		}
		/*if (hitted->GetType() == GeoType::Ball && lev == 1) {
			std::cout << "Ray Dir: " << ray.GetDir();
			std::cout << "Norm: " << norm;
			std::cout << "wi: " << wi;
			std::cout << "cosines: " << glm::dot(wi, norm) << ' ' << glm::dot(-ray.GetDir(), norm) << std::endl;
			std::cout << "Pos after hit: " << hitpos + wi * 1e-4f;
			std::cout << "dist1: " << glm::l2Norm(hitpos - glm::vec3(1, 0, 0)) << std::endl;
			std::cout << "dist2: " << glm::l2Norm(hitpos + wi * 1e-4f - glm::vec3(1, 0, 0)) << std::endl;
			std::cout << "Hitpos: " << hitpos;

		//	assert(glm::dot(wi, ray.GetDir()) < 1e-8);
		}*/
		if(lev > 4) {
			float rr = glm::clamp(fmax(coef.r, fmax(coef.g, coef.b)), 0.0f, 0.95f);
			if (lev > 30 || Random::Rand(1.0f) > rr) return coef * mat->GetGlow();	
			coef = coef / rr;
		//	return coef / rr * mat->GetGlow() +
		//		RayTracing(Ray(hitpos + wi * 1e-4f, wi), lev + 1, coef * att * fabs(glm::dot(wi, norm)) / poss) / rr;
		}
		return coef * mat->GetGlow() +
			RayTracing(Ray(hitpos + wi * 1e-4f, wi), lev + 1, coef * att * fabs(glm::dot(wi, norm)) / poss);
	}
	else return coef * m_Background;
}

void World::ComputeInfo(Geometry* geo, glm::vec3& cent, float& area) {
	switch (geo->GetType()) {
	case GeoType::Ball: {
		auto* ball = static_cast<Ball*>(geo);
		cent = ball->GetCenter();
		area = (float)(4 * pi * pow(ball->GetRadius(), 2));
		break;
	}
	case GeoType::Triangle: {
		auto* tri = static_cast<Triangle*>(geo);
		cent = (tri->GetPos(0) + tri->GetPos(1) + tri->GetPos(2)) / 3.0f;
		area = glm::l2Norm(glm::cross(tri->GetPos(1) - tri->GetPos(0), tri->GetPos(2) - tri->GetPos(0))) / 2.0f;
		break;
	}
	case GeoType::Cuboid: {
		// Do nothing
		break;
	}
	default: {
		printf("Cuboid Error: Unknown Geometry type\n");
	}
	}
}

constexpr int MAXOBJ = 10;

size_t World::DoCreateHierarchy(size_t beg, size_t ed) {
	if (ed - beg <= MAXOBJ) {
		// Only a few objects, just form a single node
		auto* cub = new Cuboid();
		std::vector<size_t> subgeos;
		for (auto obj = beg; obj < ed; ++obj) {
			subgeos.push_back(obj);
		}
		cub->AppendSubGeos(*this, subgeos);
		auto pcub = std::shared_ptr<Cuboid>(cub);
		m_Geos.push_back(Object(pcub));
		return m_Geos.size() - 1;
	}
	// Otherwise, find the best setting
	int bestdim = 0, bestpos = 0;
	float Fbest = std::numeric_limits<float>().max();
	float Stot = 0;
	for (size_t j = beg; j < ed; ++j) {
		Stot += m_Geos[j].area;
	}
	size_t n = ed - beg; // The number of elements in this section
	for (auto i = 0; i < 3; ++i) {
		// enumerate each dimensions
		// Sort the objects by dim-i
		float Sleft = 0;
		float Fmin = std::numeric_limits<float>().max(); // F = Sl*l+(Stot-Sl)*(n-l)
		size_t bpos = 0; // record the best break pos

		// Use .geo as the second parameter to make sure the sorting along the axis will always give the same outcome
		std::sort(m_Geos.begin() + beg, m_Geos.begin() + ed, [i](Object x, Object y)
			{return x.cent[i] < y.cent[i] || x.cent[i] == y.cent[i] && x.geo < y.geo;});
		for (size_t j = beg; j < ed - 1; ++j) {
			Sleft += m_Geos[j].area;
			float F = Sleft * (j - beg) + (Stot - Sleft) * (ed - j);
			if(F < Fmin) {
				Fmin = F;
				bpos = j - beg + 1;
			}
		}
		if (Fmin < Fbest) {
			Fbest = Fmin;
			bestpos = bpos;
			bestdim = i;
		}
	}
	std::sort(m_Geos.begin() + beg, m_Geos.begin() + ed, [bestdim](Object x, Object y)
		{return x.cent[bestdim] < y.cent[bestdim] || x.cent[bestdim] == y.cent[bestdim] && x.geo < y.geo;});
	auto* cub = new Cuboid();
	auto cl = DoCreateHierarchy(beg, beg + bestpos);
	auto cr = DoCreateHierarchy(beg + bestpos, ed);
	std::vector<size_t> subgeos{ cl, cr };
	cub->AppendSubGeos(*this, subgeos);
	auto pcub = std::shared_ptr<Cuboid>(cub);
	m_Geos.push_back(Object(pcub));
	return m_Geos.size() - 1;
}



void World::CreateHierarchy() {
	if (m_Root != 0) {
		// The hierarchy has already been built
		return;
	}
	m_Root = DoCreateHierarchy(0, m_Geos.size());
}
