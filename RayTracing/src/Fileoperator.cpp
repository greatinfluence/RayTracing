#include "Fileoperator.h"

namespace YAML {
	void Loadscene(std::string filepath, World& world, Image3& image) {
		YAML::Node config = YAML::LoadFile(filepath);
		Materialrepository::GetMats() = config["Mats"].as<std::vector<std::shared_ptr<Material>>>();
		world = config["World"].as<World>();
		image = config["Image"].as<Image3>();
	}

	void Savescene(std::string filepath, World const& world, Image3 const& image) {
		YAML::Node config;
		config["Mats"] = Materialrepository::GetMats();
		config["World"] = world;
		config["Image"] = image;
		std::ofstream output(filepath);
		output << config;
	}

}