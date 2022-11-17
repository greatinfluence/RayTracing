#pragma once

#include "yaml-cpp/yaml.h"
#include <iostream>
#include "la.h"
#include "Camera.h"
#include "Material.h"
#include "Triangle.h"
#include "Ball.h"
#include "Cylindsurf.h"
#include "Plate.h"
#include "World.h"
#include "Image.h"

namespace YAML {
	template<>
	struct convert<la::vec3> {
		static Node encode(la::vec3 const& rhs) {
			Node node;
			node.push_back(rhs.x);
			node.push_back(rhs.y);
			node.push_back(rhs.z);
			return node;
		}

		static bool decode(Node const& node, la::vec3& rhs) {
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
	struct convert<CamType> {
		static Node encode(CamType const& ctp) {
			return (Node)(int)ctp;
		}
		static bool decode(Node const& node, CamType& ctp) {
			if (!node.IsScalar()) return false;
			ctp = (CamType)node.as<int>();
			return true;
		}
	};

	template<>
	struct convert<std::shared_ptr<Camera>> {
		static Node encode(std::shared_ptr<Camera> const& cam) {
			Node node;
			if (cam->GetType() == CamType::Fish) {
				node["Type"] = CamType::Fish;
				FishEyeCamera* fcam = static_cast<FishEyeCamera*>(cam.get());
				node["Pos"] = fcam->m_Pos;
				node["Front"] = fcam->m_Front;
				node["Up"] = fcam->m_Up;
				node["Hor"] = fcam->m_Horang;
				node["Perp"] = fcam->m_Perang;
			}
			else if (cam->GetType() == CamType::Reg) {
				node["Type"] = CamType::Reg;
				RegularCamera* rcam = static_cast<RegularCamera*>(cam.get());
				node["Pos"] = rcam->m_Pos;
				node["Front"] = rcam->m_Front;
				node["Up"] = rcam->m_Up;
				node["Hor"] = rcam->m_Hor;
				node["Perp"] = rcam->m_Per;
			}
			return node;
		}

		static bool decode(Node const& node, std::shared_ptr<Camera>& rhs) {
			if (!node.IsMap() || node.size() != 6 || !node["Type"]) {
				return false;
			}
			if(node["Type"].as<CamType>() == CamType::Fish)
			rhs = std::shared_ptr<Camera>(new FishEyeCamera(node["Pos"].as<la::vec3>(),
				node["Front"].as<la::vec3>(), node["Up"].as<la::vec3>(),
				node["Hor"].as<double>(), node["Perp"].as<double>()));
			else if(node["Type"].as<CamType>() == CamType::Reg)
			rhs = std::shared_ptr<Camera>(new RegularCamera(node["Pos"].as<la::vec3>(),
				node["Front"].as<la::vec3>(), node["Up"].as<la::vec3>(),
				node["Hor"].as<double>(), node["Perp"].as<double>()));
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
				rhs = std::shared_ptr<Material>(new Diffuse(node["Glow"].as<la::vec3>(), node["Albedo"].as<la::vec3>()));
				return true;
			}
			case MatType::Metal: {
				rhs = std::shared_ptr<Material>(new Metal(node["Glow"].as<la::vec3>(), node["Albedo"].as<la::vec3>(), node["Fuzz"].as<float>()));
				return true;
			}
			case MatType::Dieletric: {
				rhs = std::shared_ptr<Material>(new Dieletric(node["Glow"].as<la::vec3>(), node["Ir"].as<float>()));
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
			node["Material"] = geo->m_Matid.id;
			switch (geo->GetType()) {
			case GeoType::Ball: {
				auto ball = static_cast<Ball*>(geo.get());
				node["Center"] = ball->m_Center;
				node["Radius"] = ball->m_Radius;
				break;
			}
			case GeoType::Triangle: {
				auto tri = static_cast<Triangle*>(geo.get());
				for(size_t i = 0; i < 3; ++ i)
					node["Vertices"][i] = tri->GetPos(i);
				node["Norm"] = tri->GetNorm(la::vec3(0));
				break;
			}
			case GeoType::Cylindsurf: {
				auto cyl = static_cast<Cylindsurf*>(geo.get());
				node["Center"] = cyl->m_Cent;
				node["Up"] = cyl->m_Up;
				node["Radius"] = cyl->m_Radius;
				node["Height"] = cyl->m_Height;
				break;
			}
			case GeoType::Plate: {
				auto plt = static_cast<Plate*>(geo.get());
				node["Center"] = plt->m_Cent;
				node["Up"] = plt->m_Up;
				node["Out"] = plt->m_Outrad;
				node["In"] = plt->m_Inrad;
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
				rhs = std::shared_ptr<Geometry>(new Ball(node["Center"].as<la::vec3>(), node["Radius"].as<float>()));
				break;
			}
			case GeoType::Triangle: {
				rhs = std::shared_ptr<Geometry>(new Triangle(
					node["Vertices"][0].as<la::vec3>(),
					node["Vertices"][1].as<la::vec3>(),
					node["Vertices"][2].as<la::vec3>(), node["Norm"].as<la::vec3>()));
				break;
			}
			case GeoType::Cylindsurf: {
				rhs = std::shared_ptr<Geometry>(new Cylindsurf(
					node["Center"].as<la::vec3>(),
					node["Up"].as<la::vec3>(),
					node["Radius"].as<float>(),
					node["Height"].as<float>()
				));
				break;
			}
			case GeoType::Plate: {
				rhs = std::shared_ptr<Geometry>(new Plate(
					node["Center"].as<la::vec3>(),
					node["Up"].as<la::vec3>(),
					node["Out"].as<float>(),
					node["In"].as<float>()
				));
				break;
			}
			default: {
				return false;
			}
			}
			rhs->AddMaterial(node["Material"].as<uint32_t>());
			return true;
		}
	};

	template<>
	struct convert<World> {
		static Node encode(World const& wd) {
			Node node;
			node["Camera"] = std::shared_ptr<Camera>(wd.GetCam());
			node["Background"] = wd.GetBackground();
			auto ngeo = wd.GetNgeo();
			for (auto i = 0; i < ngeo; ++i) {
				node["Geos"][i] = wd.GetGeo(i);
			}
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
			wd = World(std::shared_ptr<Camera>(node["Camera"].as<std::shared_ptr<Camera>>()), node["Background"].as<la::vec3>());
			auto geos = node["Geos"];
			for (auto it = geos.begin(); it != geos.end(); ++it) {
				wd.AddGeo(it->as<std::shared_ptr<Geometry>>());
			}
			return true;
		}
	};

	template<>
	struct convert<Image3> {
		static Node encode(Image3 const& im) {
			Node node;
			node["Width"] = im.GetWidth();
			node["Height"] = im.GetHeight();
			node["Channels"] = im.GetChannels();
			node["Filepath"] = im.GetFile();
			return node;
		}

		static bool decode(Node const& node, Image3& rhs) {
			if (!node["Width"] || !node["Height"] || !node["Channels"] || !node["Filepath"]) {
				return false;
			}
			rhs = Image3(node["Width"].as<int>(), node["Height"].as<int>(), node["Filepath"].as<std::string>(), node["Channels"].as<int>());
			return true;
		}
	};
}
