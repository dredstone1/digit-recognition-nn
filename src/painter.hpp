#ifndef PAINTER
#define PAINTER

#include <SFML/Graphics.hpp>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

class Painter {
  private:
	sf::RenderWindow window;

	std::vector<float> values;
	std::atomic<bool> enter{false};

	void reset();

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
