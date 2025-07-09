#include "src/painter.hpp"
#include <AiModel.hpp>

int main() {
	nn::AiModel model("../ModelData/config.json");

	model.train();

	App display;
	display.open();

	while (true) {
	}
	return 0;
}
