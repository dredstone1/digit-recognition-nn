#include "../include/painter.hpp"
#include <AiModel.hpp>

int main() {
	nn::AiModel model("../ModelData/config.json");
    model.load("params");
	model.train("../ModelData/data1");
	model.save("params");
	nn::model::modelResult result = model.evaluateModel("../ModelData/data");
	printf("prediction: %f\n", result.percentage);

	App display;
	display.open();

	while (display.isOpen()) {
		display.wait();

		model.runModel(display.getValues());
		printf("output: %zu, %f\n", model.getPrediction().index, model.getPrediction().value);
	}

	return 0;
}
