#include "../include/painter.hpp"
#include <AiModel.hpp>

int main() {
	nn::AiModel model("../ModelData/config.json");
	model.train("../ModelData/data1");
	model.save("params");

	App display;
	display.open();

	while (display.isOpen()) {
		display.wait();

		model.runModel(display.getValues());
		printf("output: %zu, %f\n", model.getPrediction().index, model.getPrediction().value);
	}

	return 0;
}
