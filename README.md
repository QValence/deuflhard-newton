# NLEQ Solver — Newton-Type Methods for Nonlinear Systems

## Installation

This project has minimal dependencies.

### Requirements

- Python 3.10 or newer
- `numpy`

### Install from source

Clone the repository and install the dependency in a virtual environment:

```bash
git clone https://github.com/QValence/deuflhard-newton.git
cd deuflhard-newton

python -m venv .env
source .env/bin/activate

pip install -r requirements
```

### Running the test suite

The project includes a (very) simple test suite covering basic convergence,
scaling behavior and failure modes. It can be ran with `pytest` from the project root directory:

```bash
pytest -v
```

---

## Deuflhard’s approach

The algorithm implemented here is **inspired by** the framework presented in:

> P. Deuflhard,  
> *Newton Methods for Nonlinear Problems*,  
> Springer Series in Computational Mathematics, Vol. 35, 2011.  
> DOI: 10.1007/978-3-642-23899-4

Key ideas in this framework include:

- affine-invariant formulations,
- adaptive step-size (damping) control,
- norm-based acceptance criteria,
- robustness with respect to scaling.

The code is an original implementation based on my interpretation of the underlying
mathematical principles. The focus is on algorithmic structure, not on maximum performance.
The implementation is intended for research, experimentation, and educational use, with an emphasis on clarity,
modularity, and numerical
robustness.

---

## Legal notice

- This repository contains original source code only.
- Algorithms and mathematical methods are not subject to copyright.
- The referenced book and its contents remain the property of their
  respective copyright holders.
- This project is not affiliated with, endorsed by, or approved by
  the author or the publisher.

## License

This project is licensed under the **MIT License**.

The license applies only to the source code in this repository and
does not apply to the referenced book or any other external material.
See the `LICENSE` file for details.