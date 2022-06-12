#pragma once

#include "yaml-cpp/yaml.h"
#include "glm/vec3.hpp"
#include "Camera.h"
#include "Material.h"
#include "Triangle.h"
#include "Ball.h"
#include "World.h"

namespace YAML {
	template<>
	struct convert<glm::vec3> {
		static Node encode(glm::vec3 const& rhs) {
			Node node;
			node.push_back(rhs.x);
			node.push_back(rhs.y);
			node.push_back(rhs.z);
			return node;
		}

		static bool decode(Node const& node, glm::vec3& rhs) {
			if (!node.IsSequence() || node.size() != 3) {
				return false;
			}
			rhs.x = node[0].as<float>();
			rhs.y = node[1].as<float>();
			rhs.z = node[2].as<float>();
			return true;
		}
	};

	template<>
	struct convert<Camera> {
		static Node encode(Camera const& cam) {
			Node node;
			node["Pos"] = cam.GetPos();
			node["Front"] = cam.GetFront();
			node["Up"] = cam.GetUp();
			node["Hor"] = cam.GetHor();
			node["Perp"] = cam.GetPerp();
			return node;
		}

		static bool decode(Node const& node, Camera& rhs) {
			if (!node.IsMap() || node.size() != 5) {
				return false;
			}
			rhs = Camera(node["Pos"].as<glm::vec3>(), node["Front"].as<glm::vec3>(), node["Up"].as<glm::vec3>(),
				node["Hor"].as<float>(), node["Perp"].as<float>());
			return true;
		}
	};

	template<>
	struct convert<MatType> {
		static Node encode(MatType const& mt) {
			return (Node)(int)mt;
		}
		static bool decode(Node const& node, MatType& mt) {
			if (!node.IsScalar()) return false;
			mt = (MatType)node.as<int>();
			return true;
		}
	};

	template<>
	struct convert<std::shared_ptr<Material>> {
		static Node encode(std::shared_ptr<Material> const& mat) {
			if (mat == nullptr) {
				// Empty node
				throw(std::out_of_range("YAML error: Empty pointer converting to Material"));
			}
			Node node;
			node["Type"] = mat->GetType();
			node["Glow"] = mat->GetGlow();
			switch (mat->GetType()) {
			case MatType::Diffuse: {
				auto dif = static_cast<Diffuse*>(mat.get());
				node["Albedo"] = dif->GetAlbedo();
				break;
			}
			case MatType::Metal: {
				auto met = static_cast<Metal*>(mat.get());
				node["Albedo"] = met->GetAlbedo();
				node["Fuzz"] = met->GetFuzz();
				break;
			}
			case MatType::Dieletric: {
				auto die = static_cast<Dieletric*>(mat.get());
				node["Ir"] = die->GetIr();
				break;
			}
			default: {
				throw std::out_of_range("YAML error: unrecognized Material type");
			}
			}
			return node;
		}

		static bool decode(Node const& node, std::shared_ptr<Material>& rhs) {
			if (!node["Type"]) return false;
			switch (node["Type"].as<MatType>()) {
			case MatType::Diffuse: {
				rhs = std::shared_ptr<Material>(new Diffuse(node["Glow"].as<glm::vec3>(), node["Albedo"].as<glm::vec3>()));
				return true;
			}
			case MatType::Metal: {
				rhs = std::shared_ptr<Material>(new Metal(node["Glow"].as<glm::vec3>(), node["Albedo"].as<glm::vec3>(), node["Fuzz"].as<float>()));
				return true;
			}
			case MatType::Dieletric: {
				rhs = std::shared_ptr<Material>(new Dieletric(node["Glow"].as<glm::vec3>(), node["Ir"].as<float>()));
				return true;
			}
			default:
				return false;
			}
		}
	};
	
	template<>
	struct convert<GeoType> {
		static Node encode(GeoType const& gt) {
			return (Node)(int)gt;
		}
		static bool decode(Node const& node, GeoType& gt) {
			if (!node.IsScalar()) return false;
			gt = (GeoType)node.as<int>();
			return true;
		}
	};

	template<>
	struct convert<std::shared_ptr<Geometry>> {
		static Node encode(std::shared_ptr<Geometry> const& geo) {
			if (geo == nullptr) {
				// Empty node
				throw(std::out_of_range("YAML error: Empty pointer converting to Geometry"));
			}
			Node node;
			node["Type"] = geo->GetType();
			node["Material"] = geo->GetMaterial();
			switch (geo->GetType()) {
			case GeoType::Ball: {
				auto ball = static_cast<Ball*>(geo.get());
				node["Center"] = ball->GetCenter();
				node["Radius"] = ball->GetRadius();
				break;
			}
			case GeoType::Triangle: {
				auto tri = static_cast<Triangle*>(geo.get());
				for(size_t i = 0; i < 3; ++ i)
					node["Vertices"][i] = tri->GetPos(i);
				node["Norm"] = tri->GetNorm(glm::vec3(0));
				break;
			}
			default: {
				throw std::out_of_range("YAML error: unrecognized Geometry type");
			}
			}
			return node;
		}

		static bool decode(Node const& node, std::shared_ptr<Geometry>& rhs) {
			if (!node["Type"]) return false;
			if (!node["Material"]) {
				std::cout << "No mat" << std::endl;
				return false;
			}
			switch (node["Type"].as<GeoType>()) {
			case GeoType::Ball: {
				rhs = std::shared_ptr<Geometry>(new Ball(node["Center"].as<glm::vec3>(), node["Radius"].as<float>()));
				break;
			}
			case GeoType::Triangle: {
				rhs = std::shared_ptr<Geometry>(new Triangle(
					node["Vertices"][0].as<glm::vec3>(),
					node["Vertices"][1].as<glm::vec3>(),
					node["Vertices"][2].as<glm::vec3>(), node["Norm"].as<glm::vec3>()));
				break;
			}
			default: {
				return false;
			}
			}
			rhs->AddMaterial(node["Material"].as<std::shared_ptr<Material>>());
			return true;
		}
	};

	template<>
	struct convert<World> {
		static Node encode(World const& wd) {
			Node node;
			node["Camera"] = wd.GetCam();
			node["Background"] = wd.GetBackground();
			node["Geos"] = wd.GetGeos();
			return node;
		}
		
		static bool decode(Node const& node, World& wd) {
			if (!node["Background"]) {
				std::cout << "No Bg" << std::endl;
				return false;
			}
			if (!node["Camera"]) {
				std::cout << "No Camera" << std::endl;
				return false;
			}
			if (!node["Geos"]) {
				std::cout << "No Geos" << std::endl;
				return false;
			}
			if (!node["Camera"] || !node["Background"] || !node["Geos"]) return false;
			wd = World(node["Camera"].as<Camera>(), node["Background"].as<glm::vec3>());
			auto geos = node["Geos"];
			for (auto it = geos.begin(); it != geos.end(); ++it) {
				wd.AddGeo(it->as<std::shared_ptr<Geometry>>());
			}
			return true;
		}
	};
}