MAKEFLAGS = -j$(nproc)
CC = g++
CFLAGS = -std=c++17 -pthread -O -Wno-comment

.PHONY: no-target
no-target:
	@echo "\033[91mError\033[0m: Please select one of these targets: make Linux, make macOS, make Android"

Linux macOS Android: LDFLAGS_OPENCL = -I./src/OpenCL/include

Linux: LDLIBS_OPENCL = -L./src/OpenCL/lib -lOpenCL
macOS: LDLIBS_OPENCL = -framework OpenCL
Android: LDLIBS_OPENCL = -L/system/vendor/lib64 -lOpenCL

Linux macOS Android: bin/FluidX3D

bin/FluidX3D: temp/cli.o temp/fluidx3d.o temp/info.o temp/kernel.o temp/lbm.o temp/main.o temp/setup.o temp/shapes.o make.sh
	@mkdir -p bin
	$(CC) temp/*.o -o bin/FluidX3D $(CFLAGS) $(LDFLAGS_OPENCL) $(LDLIBS_OPENCL)

temp/cli.o: src/cli.cpp src/fluidx3d.h src/defines.hpp src/lbm.hpp src/info.hpp src/setup.hpp make.sh
	@mkdir -p temp
	$(CC) -c src/cli.cpp -o temp/cli.o $(CFLAGS) $(LDFLAGS_OPENCL)

temp/fluidx3d.o: src/fluidx3d.cpp src/fluidx3d.h src/config.hpp src/defines.hpp src/lbm.hpp make.sh
	@mkdir -p temp
	$(CC) -c src/fluidx3d.cpp -o temp/fluidx3d.o $(CFLAGS) $(LDFLAGS_OPENCL)

temp/info.o: src/info.cpp src/defines.hpp src/config.hpp src/info.hpp src/lbm.hpp src/opencl.hpp src/units.hpp src/utilities.hpp make.sh
	@mkdir -p temp
	$(CC) -c src/info.cpp -o temp/info.o $(CFLAGS) $(LDFLAGS_OPENCL)

temp/kernel.o: src/kernel.cpp src/kernel.hpp src/utilities.hpp
	@mkdir -p temp
	$(CC) -c src/kernel.cpp -o temp/kernel.o $(CFLAGS)

temp/lbm.o: src/lbm.cpp src/defines.hpp src/config.hpp src/info.hpp src/lbm.hpp src/opencl.hpp src/units.hpp src/utilities.hpp make.sh
	@mkdir -p temp
	$(CC) -c src/lbm.cpp -o temp/lbm.o $(CFLAGS) $(LDFLAGS_OPENCL)

temp/main.o: src/main.cpp src/defines.hpp src/info.hpp src/lbm.hpp src/opencl.hpp src/setup.hpp src/shapes.hpp src/units.hpp src/utilities.hpp make.sh
	@mkdir -p temp
	$(CC) -c src/main.cpp -o temp/main.o $(CFLAGS) $(LDFLAGS_OPENCL)

temp/setup.o: src/setup.cpp src/defines.hpp src/config.hpp src/info.hpp src/lbm.hpp src/opencl.hpp src/setup.hpp src/shapes.hpp src/units.hpp src/utilities.hpp make.sh
	@mkdir -p temp
	$(CC) -c src/setup.cpp -o temp/setup.o $(CFLAGS) $(LDFLAGS_OPENCL)

temp/shapes.o: src/shapes.cpp src/shapes.hpp src/utilities.hpp make.sh
	@mkdir -p temp
	$(CC) -c src/shapes.cpp -o temp/shapes.o $(CFLAGS) $(LDFLAGS_OPENCL)

.PHONY: clean
clean:
	@rm -rf temp bin/FluidX3D
