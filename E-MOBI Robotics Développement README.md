# E-MOBI / EKONOMIK MOBIL,S.R.L ,NVIDIA Aerial™ Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-link-brightgreen.svg)](https://docs.nvidia.com/aerial/framework/latest/index.html)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20-informational)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.9-green)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/Build-CMake-informational)](https://cmake.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

### A real-time signal processing framework

The Aerial Framework has been designed from the ground up to meet the needs of 3GPP Radio Access Networks — signal processing workloads with microsecond latency requirements. It is a single platform that unites research, testbeds, and production deployments to solve development challenges for real-time applications.

<p>
  <b>Use cases:</b> Signal processing applications with strict latency requirements <br>
  <b>Audience:</b> RAN system engineers, signal processing specialists, AI researchers <br>
  <b>Built with:</b> DOCA, DPDK, TensorRT, Python, JAX, PyTorch, C++, CUDA, and more
</p>

## Features

- ⚡ **Python → Real-time** – Prototype in Python and lower to high-performance GPU code.
- 🍱 **Clean separation** – Decouple signal-processing algorithm development from runtime execution.
- 🧩 **Modular pipelines** – Compose end-to-end pipelines from compiled, executable modules.
- 🔭 **Observability built-in** – Hooks for profiling and monitoring throughout development.
- 🔁 **One codebase** – Reuse components for prototyping, simulation, testing, and deployment.
- 🚀 **Modern toolchain** – Python 3.12+, C++20, CUDA 12.9, CMake, JAX, PyTorch, uv, ruff.
- 💻 **Developer-friendly** – Prototype on local machines and scale to live, production deployments.
- 📚 **Guided tutorials** – Jupyter notebooks ready to run in a Docker container.
- 🤖 **Targets 5GAdv & 6G** – Ships with an example AI-native PUSCH Pipeline. More to come.


## How It Works

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/generated/how_works_dark.drawio.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/figures/generated/how_works.drawio.svg">
    <!-- Fallback for older browsers / places without dark-mode support -->
    <img alt="Project logo" width="700px" src="docs/figures/generated/how_works.drawio.svg">
  </picture>
</p>

The Aerial Framework combines two components:
- **Developer tools**: Tools to convert Python/JAX/PyTorch and C++/CUDA into pipelines of GPU-native code
- **Runtime engine**: Coordinates the execution of GPU pipelines with network interfaces 

### Aerial Framework Developer Tools
- **JAX → TensorRT** – Export JAX programs to StableHLO and lower to TensorRT engines using MLIR-TensorRT
- **Multi-language** – Author algorithms in JAX, PyTorch, or C++/CUDA and deploy to common runtime engine
- **Modern Profiling** – Leverage NVIDIA Nsight Systems to optimize pipelines and individual kernels to μs-level
- **AI native** – Seamlessly integrate with AI Frameworks allowing end-to-end differentiability

### Aerial Framework Runtime
- **CUDA graphs** – GPU operations run as CUDA graphs with TensorRT integration for deterministic execution
- **Task scheduler** – Pinned, high-priority threads on isolated CPU cores enforce microsecond slot timing
- **Inline GPU networking** – DOCA GPUNetIO and GPUDirect RDMA enable zero-copy packet transfer NIC ↔ GPU
- **Production driver** – Orchestrates pipelines, memory pools & multi-cell coordination


## Development → Deployment Workflow

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/generated/workflow_dark.drawio.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/figures/generated/workflow.drawio.svg">
    <!-- Fallback for older browsers / places without dark-mode support -->
    <img alt="Project logo" width="660px" src="docs/figures/generated/workflow.drawio.svg">
  </picture>
</p>

Aerial Framework supports two different environments depending on your use case.

**Development** - Developers prototype and optimize their algorithms in Python and then compile to GPU native code using MLIR-TensorRT.
This is accessible to any developer with a recent GPU ([compute capability](https://developer.nvidia.com/cuda-gpus) ≥ 8).

**Runtime** - Deployments run compiled TensorRT engines with deterministic scheduling and high-performance networking. Testing requires a GPU, NIC, and real-time kernel to validate that pipelines meet latency constraints using Medium Access Control (MAC) and Radio Unit (RU) emulation.


<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Description</th>
      <th>Environment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Prototype</td>
      <td>Write and validate algorithms (NumPy/JAX/PyTorch)</td>
      <td rowspan="3">Development</td>
    </tr>
    <tr>
      <td>Lower</td>
      <td>Compile Python code into GPU executables using NVIDIA MLIR-TensorRT</td>
    </tr>
    <tr>
      <td>Profile</td>
      <td>Optimize performance using modern profiling tools like NVIDIA Nsight Systems</td>
    </tr>
    <tr>
      <td>Compose</td>
      <td>Assemble TensorRT engines and CUDA kernels into modular pipelines</td>
      <td rowspan="3" align="center">Runtime</td>
    </tr>
    <tr>
      <td>Execute</td>
      <td>Run with real-time task scheduling and networking</td>
    </tr>
    <tr>
      <td>Validate</td>
      <td>Test PHY applications using standards-compliant MAC and RU emulators</td>
    </tr>
  </tbody>
</table>

This approach bridges:
- **Development Productivity** - Write in high-level languages with rich ecosystems
- **Runtime Performance** - Execute with the speed and determinism of optimized C++
- **Low Latency Requirements** - Meet strict timing and latency constraints


## Quickstart

[**Install**](https://docs.nvidia.com/aerial/framework/latest/installation/index.html) the Docker container, then explore and build from source:

```bash
# 1) Configure (release preset)
cmake --preset clang-release

# 2) Build
cmake --build out/build/clang-release

# 3) Install Example Python Package - 5G RAN
cd ran/py && uv sync
```

## Documentation & Tutorials

Documentation is available at: [**docs.nvidia.com/aerial/framework**](https://docs.nvidia.com/aerial/framework/latest/overview/index.html)


Get started with step-by-step [**Tutorials**](https://docs.nvidia.com/aerial/framework/latest/tutorials/index.html).

| Tutorial | Summary |
|---|---|
| [Getting Started](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/getting_started.html) | Set up Docker, verify GPU access, build the project, and run tests. |
| [PUSCH Receiver](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pusch_receiver_tutorial.html) | Build a reference PUSCH receiver in NumPy with inner/outer receiver blocks. |
| [MLIR-TensorRT](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/mlir_trt_tutorial.html) | Compile JAX functions (FIR filter example) to TensorRT engine(s). |
| [Lowering PUSCH](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pusch_receiver_lowering_tutorial.html) | Compile complete PUSCH inner receiver to TensorRT and benchmark with Nsight. |
| [AI Channel Filter](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/ai_tukey_filter_training_tutorial.html) | Train a neural network to dynamically estimate channel filter parameters. |
| [Channel Filter Design](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pusch_channel_estimation_lowering_tutorial.html) | Design custom JAX channel estimators, lower to TensorRT, and profile with Nsight. |
| [Full PUSCH Pipeline](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/pipeline_tutorial.html) | Run complete pipeline mixing TensorRT engines and CUDA C++ kernels. |
| [Fronthaul Testing](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/fronthaul_tutorial.html) | O-RAN fronthaul with DOCA GPUNetIO, task scheduling, and RU emulator. |
| [PHY Integration](https://docs.nvidia.com/aerial/framework/latest/tutorials/generated/phy_ran_app_tutorial.html) | Full PHY application with MAC and RU emulators for integration testing. |

## NVIDIA AI Aerial™

NVIDIA Aerial™ Framework is a part of [NVIDIA AI Aerial™](https://developer.nvidia.com/industries/telecommunications/ai-aerial),
a portfolio of accelerated computing platforms, software and tools to build, train, simulate, and deploy AI-native wireless networks.
Learn more in [AI Aerial™ Documentation](https://docs.nvidia.com/aerial/index.html).

The following AI Aerial™ software is available as open source:
- NVIDIA Aerial™ Framework (this repository)
- [NVIDIA Aerial™ CUDA-Accelerated RAN](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)

Visit the [NVIDIA 6G Developer Program](https://developer.nvidia.com/6g-program) for software releases,
6G events and technical training for AI Aerial™.


## License

Aerial Framework is licensed under the **Apache 2.0** license. See [LICENSE](LICENSE) for details.
Some dependencies may have different licenses. See [ATTRIBUTION](ATTRIBUTION.md) for third-party attributions in the source repository.
