# Queue Reactive Model Implementation

Limit order book simulation framework based on the Queue Reactive (QR) model.

## Requirements

- **C++20** (Compiler I used gcc13.3.0)
- **CMake** 3.15+
- **vcpkg** (bundled as submodule)

## Build

```bash
# Clone with submodules
git clone --recursive https://github.com/youruser/Alpha-Queue-Reactive.git
cd qr

# Bootstrap vcpkg (first time only)
./vcpkg/bootstrap-vcpkg.sh

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build -j$(nproc)
```

## Project Structure

```
qr/
├── cpp/
│   ├── include/        # Headers
│   ├── src/            # Implementation
│   └── bin/            # Executables (simulation loops)
├── src/qr/             # Python package
├── notebooks/          # Jupyter notebooks
└── data/               # Input data
```
