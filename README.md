# Limit Order Book Simulation

Implementation of [Bridging the Reality Gap in Limit Order Book Simulation](https://arxiv.org/abs/2603.24137)

> Microstructure and data driven Limit order book simulation based on the [queue reactive model](https://arxiv.org/pdf/1312.0563).

A C++ simulation framework implementing the **Queue-Reactive (QR) model** for large-tick assets. The QR model treats the limit order book as a continuous-time Markov jump process: every event (limit order, cancellation, trade) is sampled from empirical distributions conditioned on the current book state — volume imbalance and spread.

The framework pairs with a **Python estimation pipeline** that calibrates all parameters from [Databento](https://databento.com) MBP-10 market data, plus Jupyter notebooks for analysis and paper figures.

---

### Requirements

| Component | Version |
|-----------|---------|
| C++ | C++20 (gcc 13.3.0) |
| CMake | 3.15+ |
| vcpkg | bundled as submodule |
| Python | 3.11+ with `uv` |

### Build

The easiest way to build is inside a container using [Podman](https://podman.io/) (or Docker — the `Containerfile` is compatible with both):

```bash
git clone --recurse-submodules https://github.com/SaadSouilmi/Queue-Reactive.git
cd Queue-Reactive
git submodule update --init --recursive

podman build -t qr .
podman run --rm -it qr bash
```

This handles all C++ and Python dependencies automatically.

### Project Structure

```
qr/
├── cpp/
│   ├── include/          # Headers (orderbook, qr_model, simulation, strategy)
│   ├── src/              # Implementation
│   └── bin/              # Executables (sample, sample_strategy, run_metaorder, ...)
├── src/qr/               # Python estimation & analysis package
├── scripts/              # Plotting scripts
├── notebooks/            # Jupyter notebooks (paper figures, calibration)
├── configs/              # JSON simulation configs
├── tests/                # C++ unit tests (GoogleTest)
├── data/                 # Input data & simulation results
└── docs/                 # Documentation site (Quarto)
```

### Documentation

Further details available [here](https://saadsouilmi.github.io/Queue-Reactive/) covering:

- **[Model and Data](https://saadsouilmi.github.io/Queue-Reactive/model.html)** — the QR framework, estimation pipeline, and probability biasing
- **[Order Book](https://saadsouilmi.github.io/Queue-Reactive/orderbook.html)** — book representation, event types, and C++ implementation
- **[Simulation Loop](https://saadsouilmi.github.io/Queue-Reactive/simulation.html)** — configuring and running simulations

### Citation

```bibtex
@misc{noble2026bridgingrealitygaplimit,
      title={Bridging the Reality Gap in Limit Order Book Simulation},
      author={Patrick Noble and Mathieu Rosenbaum and Saad Souilmi},
      year={2026},
      eprint={2603.24137},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2603.24137},
}
```
