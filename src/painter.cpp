#include "painter.hpp"

Painter::Painter()
    : window(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}), "SFML Window"),
      values(GRID_SIZE * GRID_SIZE) {
}

void Painter::open() {
	while (window.isOpen()) {
		window.clear(BG_COLOR);

		renderCanvas();

		window.display();
	}

	window.close();
}

void Painter::renderCanvas() {
	const float cellSize = (CANVAS_SIZE - (GRID_SIZE - 1) * PIXEL_GAP) / GRID_SIZE;

	sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
	values[4] = 1;
	int currentPixel = 0;
	for (int y = 0; y < GRID_SIZE; ++y) {
		for (int x = 0; x < GRID_SIZE; ++x) {
			float posX = UI_GAP + x * (cellSize + PIXEL_GAP);
			float posY = UI_GAP + y * (cellSize + PIXEL_GAP);

			cell.setPosition({posX, posY});

			int currentValue = values[currentPixel];
			cell.setFillColor(sf::Color(currentValue, currentValue, currentValue));

			window.draw(cell);

			currentPixel++;
		}
	}
}
void Painter::reset() {
	for (auto &value : values) {
		value = 0;
	}
}

void Painter::cancleEnter() {
	enter = false;
}

bool Painter::checkEnter() const {
	return enter;
}

App::App() {
}

void App::open() {
	if (running)
		return;

	displayThread = std::thread(&App::start, this);
	while (!running) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

void App::start() {
	if (painter)
		return;

	painter = std::make_unique<Painter>();
	running = true;

	painter->open();
}

void App::wait() {
	if (!running) {
		return;
	}

	while (!painter->checkEnter()) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	painter->cancleEnter();
}

const std::vector<float> &App::getValues() {
	if (running) {
		return painter->getValues();
	}

	static const std::vector<float> empty;
	return empty;
}
