# EdgeSplit-Bench node image: llama.cpp (with RPC backend) + network tools.
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
ARG LLAMA_CPP_REF=b6000
RUN git checkout ${LLAMA_CPP_REF} \
    && echo "${LLAMA_CPP_REF} $(git rev-parse HEAD)" > /llama_cpp_commit.txt
RUN cmake -B build -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release -j"$(nproc)" --target llama-bench llama-cli
RUN T="$(cmake --build build --target help | grep -oE '(llama-)?rpc-server' | head -n1)"; \
    if [ -z "$T" ]; then echo "ERROR: no rpc-server target found. RPC targets:"; \
      cmake --build build --target help | grep -i rpc; exit 1; fi; \
    echo "building target: $T"; \
    cmake --build build --config Release -j"$(nproc)" --target "$T"; \
    mkdir -p /out; cp build/bin/llama-bench build/bin/llama-cli /out/; cp "build/bin/$T" /out/rpc-server

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 iproute2 iputils-ping iperf3 procps bash && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /usr/local/bin/
COPY --from=build /llama_cpp_commit.txt /llama_cpp_commit.txt
CMD ["sleep", "infinity"]
