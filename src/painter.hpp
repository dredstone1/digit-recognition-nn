#ifndef PAINTER
#define PAINTER

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

constexpr sf::Color BG_COLOR(100, 100, 100);
constexpr sf::Color PANELS_BG(255, 255, 255);

constexpr std::uint32_t WINDOW_HEIGHT = 600;
constexpr std::uint32_t WINDOW_WIDTH = 1000;

constexpr std::uint32_t UI_GAP = 15;
constexpr float PIXEL_GAP = 1;

constexpr std::uint32_t CANVAS_SIZE = WINDOW_HEIGHT - UI_GAP * 2;

constexpr int GRID_SIZE = 28;


class Painter {
  private:
	sf::RenderWindow window;

	std::vector<float> values;
	std::atomic<bool> enter{false};

	void reset();

	void renderCanvas();

  public:
	Painter();
	~Painter() = default;

	void open();
	bool checkEnter() const;
	void cancleEnter();
	const std::vector<float> &getValues() { return values; }
};

class App {
  private:
	std::thread displayThread;
	std::unique_ptr<Painter> painter;
	std::atomic<bool> running{false};

	void start();

  public:
	App();
	~App() = default;

	void open();
	void wait();
	const std::vector<float> &getValues();
};

#endif // PAINTER
