#include "src/painter.hpp"
#include <AiModel.hpp>

int main() {
	nn::AiModel model("");

	model.train();

	painter display;

	display.open();

	return 0;
}
