# moefast-35B — build from source in a CUDA image (no per-arch binaries needed)
# Usage:
#   docker build -f moefast/Dockerfile -t moefast:cu12 .
#   docker run --gpus all -p 8080:8080 -v /models:/models moefast:cu12 /models/model.gguf
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
 && cmake --build build --target llama-server -j"$(nproc)"

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/bin/llama-server /app/llama-server
COPY --from=build /src/build/bin/*.so* /app/
COPY moefast/heat/heat_qwen36.bin /app/heat/heat_qwen36.bin
COPY moefast/run.sh /app/run.sh
ENV LD_LIBRARY_PATH=/app
ENV LLAMA_EXPERT_HEAT=/app/heat/heat_qwen36.bin
EXPOSE 8080
ENTRYPOINT ["/app/llama-server"]
CMD ["--help"]
