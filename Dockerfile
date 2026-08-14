FROM sdukshis/conangcc:11.2

ARG BUILD_TYPE=Release

WORKDIR /workspace

RUN apt-get update && \
    apt-get install --yes --no-install-recommends libeigen3-dev libgtest-dev

COPY CMakeLists.txt ./
COPY src/ src/
COPY tests/ tests/
COPY data/ data/
COPY model/ model/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
RUN cmake --build build
