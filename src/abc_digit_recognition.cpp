#include "../include/painter.hpp"
#include "../include/transformation.hpp"
#include "consts.hpp"
#include "dataBase.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <model.hpp>

const int TOP_X = 5;

static App display;
static bool isOpen = false;

constexpr bool SHOW_LIVE_T = false;
constexpr size_t GRID_LEN = GRID_SIZE * GRID_SIZE;

void move(const nn::global::Tensor &p, nn::global::Tensor &result);

std::vector<nn::global::ValueType> data(GRID_LEN);

class dbt : public nn::model::DataBase {
  public:
	nn::model::TrainSample getSample(const size_t i) override {
		nn::model::TrainSample newSample(samples.samples[i]);

		if (SHOW_LIVE_T) {
			if (!isOpen)
				display.open();
			isOpen = true;
			newSample.input.getData(data);
			display.setValues(data);
			std::this_thread::sleep_for(std::chrono::seconds(3)); // 3 seconds
		}

		if (nn::global::Tensor::getGpuState()) {
			move(samples.samples[i].input, newSample.input);

		} else {
			tr::box newBox = tr::getBox(samples.samples[i].input);

			int v = tr::getAction(-newBox.x, 28 - (newBox.width + newBox.x));
			int h = tr::getAction(-newBox.y, 28 - (newBox.height + newBox.y));

			tr::move(newSample.input, newBox, h, v);
		}

		if (SHOW_LIVE_T) {
			std::this_thread::sleep_for(std::chrono::seconds(3)); // 3 seconds
			newSample.input.getData(data);
			display.setValues(data);
		}
		return newSample;
	}
};

int main(int argc, char *argv[]) {
	nn::global::Tensor::toGpu();
	nn::model::Model model("../ModelData/emnist_config.json");

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-l") {
			std::cout << "loading command emnist\n";
			model.load("emnist_model.txt");
		} else if (arg == "-t") {
			std::cout << "training command emnist\n";
			std::vector<std::string> files{
			    "../ModelData/emnist_letters_train"};

			dbt db;
			db.load(files);
			dbt db1;
			db1.load({"../ModelData/emnist_letters_test"});

			model.train(db, db1);
			model.save("emnist_model.txt");
		}
	}

	std::vector<std::string> files{
	    "../ModelData/emnist_letters_test"};
	nn::model::DataBase db2;
	db2.load(files);
	nn::model::modelResult result = model.evaluateModel(db2);
	printf("final score evaluation: %f\n", result.percentage);
	if (!isOpen)
		display.open();
	isOpen = true;

	while (display.isOpen()) {
		display.wait();
		nn::global::Tensor metrix({784});
		metrix = display.getValues();

		model.runModel(metrix);
		std::vector<nn::global::ValueType> output = model.getOut();

		// Create vector of pairs (index, value)
		std::vector<std::pair<size_t, nn::global::ValueType>> indexedOutput;
		for (size_t i = 0; i < output.size(); ++i) {
			indexedOutput.push_back({i, output[i]});
		}

		// Sort descending by value
		std::sort(indexedOutput.begin(), indexedOutput.end(),
		          [](const auto &a, const auto &b) { return a.second > b.second; });

		// Print top X in one line
		for (int i = 0; i < std::min(TOP_X, (int)indexedOutput.size()); ++i) {
			size_t idx = indexedOutput[i].first;
			nn::global::ValueType val = indexedOutput[i].second;
			std::string character = emnist_balanced[idx].name;
			std::cout << character << ":" << val;
			if (i != std::min(TOP_X, (int)indexedOutput.size()) - 1)
				std::cout << " | ";
		}
		std::cout << std::endl;
	}

	return 0;
}
