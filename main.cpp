#include "src/painter.hpp"
#include <AiModel.hpp>

int main() {
	nn::AiModel model("../ModelData/config.json");
	//
	// model.train();

	App display;
	display.open();

	while (true) {
		display.wait();

		model.runModel(display.getValues());
		printf("output: %d, %f\n", model.getPrediction().index, model.getPrediction().value);
	}
	return 0;
}
