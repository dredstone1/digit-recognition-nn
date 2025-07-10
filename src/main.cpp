#include "../include/painter.hpp"
#include <AiModel.hpp>

int main() {
	nn::AiModel model("../ModelData/config.json");
	model.train();
    model.save("params");

	App display;
	display.open();

	while (display.isOpen()) {
		display.wait();

		model.runModel(display.getValues());
		printf("output: %d, %f\n", model.getPrediction().index, model.getPrediction().value);
	}
	return 0;
}
