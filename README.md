# Next Generation Reservoir Computing (NG‑RC) for Nonlinear Dynamic Systems Analysis

This project was developed as part of my Master's thesis at COPPE/UFRJ under the supervision of Thiago Gamboa Ritto. The thesis, titled "Next Generation Reservoir Computing (NG-RC) for the analysis of nonlinear dynamic systems" explores the application of NG‑RC to model and predict the behavior of complex nonlinear dynamic systems.

## Project Overview

This project focuses on two main case studies:
1. **Double Pendulum Analysis:**  
   Simulation and prediction of a double pendulum’s chaotic dynamics under various initial conditions, including experiments with extra parameters.
2. **Drill String (TLDS) Analysis:**  
   Modeling and prediction of lateral-torsional vibrations in drill string systems using the base NG‑RC framework with specialized data preprocessing.

## Repository Structure

- **README.md**: Project overview and usage instructions.
- **.gitignore**: Git ignore file (ignores `__pycache__` and `.ipynb_checkpoints`).
- **requirements.txt**: List of project dependencies.
- **src/**
  - **models/**
    - **double_pendulum.py**: Double pendulum simulation model.
    - **drill_string.py**: Drill string (TLDS) simulation model.
  - **ngrc/**
    - **ngrc_model.py**: NG‑RC model implementation (feature construction, training, prediction, and metrics).
  - **utils/**  
    - **solver.py**: ODE solver wrapper for simulating dynamic systems.
    - **plot_helper.py**: Plotting utilities for visualizing simulation and prediction results.
- **notebooks/**
  - **double_pendulum.ipynb**: Base NG‑RC analysis for the double pendulum.
  - **double_pendulum_with_params.ipynb**: Double pendulum analysis using NG‑RC with additional parameters.
  - **double_pendulum_multi_initial_conditions.ipynb**: Double pendulum analysis using NG‑RC with multiple initial conditions.
  - **drill_string.ipynb**: NG‑RC analysis for the drill string system.

## Installation

Ensure you have Python 3.x installed. Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

To run the experiments, open and execute the notebooks in the `notebooks` folder:
- **double_pendulum.ipynb**: Base NG‑RC analysis for the double pendulum.
- **double_pendulum_with_params.ipynb**: Double pendulum analysis using NG‑RC with additional parameters.
- **double_pendulum_multi_initial_conditions.ipynb**: Double pendulum analysis using NG‑RC with multiple initial conditions.
- **drill_string.ipynb**: NG‑RC analysis for the drill string system.

The modular code in the `src` directory can also be imported into your own scripts for further experimentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For inquiries, please contact:

**Higor Rodrigues Paixão**  
Email: [higor.rodriguesss@hotmail.com]

