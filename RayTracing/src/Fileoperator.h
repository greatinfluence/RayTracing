#pragma once

#include <string>

#include "Yamlext.h"
#include "Materialrepository.h"

#include <fstream>

namespace YAML {
	void Loadscene(std::string filepath, World& world, Image3& image);

	void Savescene(std::string filepath, World const& world, Image3 const& image);
}
