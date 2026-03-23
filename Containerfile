FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    zip \
    unzip \
    tar \
    pkg-config \
    ninja-build \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy vcpkg manifest first for caching
COPY vcpkg.json .

# Copy vcpkg submodule
COPY vcpkg/ vcpkg/
RUN ./vcpkg/bootstrap-vcpkg.sh

# Install vcpkg dependencies (cached unless vcpkg.json changes)
RUN ./vcpkg/vcpkg install

# Copy the rest of the project
COPY . .

# Build C++
RUN cmake -B build \
    -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc)

# Install Python dependencies
RUN uv sync

# Run tests to verify
RUN cd build && ctest --output-on-failure

CMD ["bash"]
