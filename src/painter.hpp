#ifndef PAINTER
#define PAINTER

#include <SFML/System/Vector2.hpp>
#include <vector>

class painter {
	std::vector<float> values;

  public:
	painter();
	~painter() = default;

	void open();
	void reset();
	std::vector<float> &getValues() { return values; };
};

#endif // PAINTER
