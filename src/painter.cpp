#include "../include/painter.hpp"

Painter::Painter()
    : window(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}), "SFML Window"),
      values(GRID_SIZE * GRID_SIZE) {
}

void Painter::open() {
	while (window.isOpen() && running) {
		window.clear(BG_COLOR);

		processEvents();
		renderCanvas();

		window.display();
	}

	window.close();
}

void Painter::close() {
	running = false;
}

void Painter::processEvents() {
	while (const std::optional event = window.pollEvent()) {
		if (event->is<sf::Event::Closed>()) {
			running = false;
		} else if (const auto *mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
			if (mouseButtonPressed->button == sf::Mouse::Button::Right) {
				mouseActive = MouseMode::remove;
			} else {
				mouseActive = MouseMode::paint;
			}
		} else if (event->is<sf::Event::MouseButtonReleased>()) {
			mouseActive = MouseMode::none;
		} else if (const auto *keyPressed = event->getIf<sf::Event::KeyPressed>()) {
			if (keyPressed->scancode == sf::Keyboard::Scancode::Enter) {
				enter = true;
			} else if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
				reset();
			}
		}
	}
}

void Painter::renderCanvas() {
	sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));

	int currentPixel = 0;
	for (int y = 0; y < GRID_SIZE; ++y) {
		for (int x = 0; x < GRID_SIZE; ++x) {
			cell.setPosition({UI_GAP + x * (CELL_SIZE + PIXEL_GAP),
			                  UI_GAP + y * (CELL_SIZE + PIXEL_GAP)});

			if (mouseActive != MouseMode::none &&
			    cell.getGlobalBounds().contains(sf::Vector2f(sf::Mouse::getPosition(window)))) {

				if (mouseActive == MouseMode::paint) {
					values[currentPixel] = 255;
				} else {
					values[currentPixel] = 0;
				}
			}

			auto currentValue = values[currentPixel];
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

void App::close() {
	running = false;

	if (displayThread.joinable()) {
		if (painter) {
			painter->close();
		}

		displayThread.join();
	}
}

void App::wait() {
	if (!running) {
		return;
	}

	while (!painter->checkEnter() && running) {
		if (!painter->isRunning()) {
			close();
		}
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
