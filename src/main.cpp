#include "info.hpp"
#include "lbm.hpp"
#include "setup.hpp"

vector<string> main_arguments;
std::atomic_bool running = true;

void main_physics() {
	info.print_logo();
	main_setup(); // execute setup
	running = false;
	exit(0); // make sure that the program stops
}

// main() is now in cli.cpp
