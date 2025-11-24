# Deep Dive: ANN for Hybrid Nanofluid Flow


## 🧐 What Are We Doing Here?

We're simulating the behavior of **hybrid nanofluids** (specifically Copper-Alumina nanoparticles suspended in water) flowing over a stretching sheet. This is a classic problem in fluid dynamics with real-world applications in:
- **Cooling systems** for electronics and industrial processes
- **Solar energy** collectors and thermal management
- **Heat exchangers** and thermal engineering
- **Biomedical devices** and drug delivery systems

### The Traditional Approach

Usually, you'd solve this using numerical methods like the shooting method or `solve_bvp` from SciPy. While these methods are accurate, they have some drawbacks:
- **Slow**: Each parameter combination requires solving complex differential equations
- **Computationally heavy**: Not suitable for real-time applications
- **Not scalable**: Running thousands of simulations takes significant time

### Our Machine Learning Approach

**Our Goal:** Create a neural network that can:
1. **Learn** the complex physics from numerical data
2. **Predict** flow velocity (f') and temperature (θ) profiles instantly
3. **Match** the accuracy of traditional numerical solvers
4. **Generalize** to new parameter combinations within the training range

We use a **9-layer Deep Neural Network** trained with the **Levenberg-Marquardt** algorithm and **Adam optimizer** to achieve this. It's not just a "black box"—we've rigorously validated it to ensure it respects the underlying physics.

---

## 🏗️ Project Structure

Here's how we organized the code. We believe in keeping things modular and clean:

```
ANN-net/
├── docs/                      # Documentation (you are here!)
│   ├── README.md              # This detailed guide
│   ├── ARCHITECTURE.md        # Deep dive into the Neural Net design
│   ├── USAGE.md               # Step-by-step usage instructions
│   └── MANUSCRIPT.docx        # Research manuscript draft
├── src/                       # Source code
│   ├── config.py              # Central configuration (parameters & settings)
│   ├── main.py                # Command-line interface and pipeline orchestrator
│   ├── generate_data.py       # Physics-based dataset generator
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── trainer.py             # Training loop and optimization
│   ├── visualizer.py          # Publication-quality plotting
│   ├── validate_model.py      # Model validation and error analysis
│   ├── models/                # Neural network definitions
│   │   ├── ann.py             # ANN architecture
│   │   └── lm_optimizer.py    # Levenberg-Marquardt optimizer
│   └── solver/                # Numerical ODE solvers
│       └── ode_solver.py      # Boundary value problem solver
├── data/                      # Generated datasets (created on first run)
│   ├── training_data.csv      # Training dataset (~32,400 samples)
│   └── test_data.csv          # Test dataset (10 random cases)
├── outputs/                   # Generated outputs
│   ├── models/                # Trained models and scalers
│   └── plots/                 # Visualizations and validation plots
├── requirements.txt           # Python dependencies
└── README.md                  # Quick start guide
```

---

## 💻 Installation

### Prerequisites
You'll need **Python 3.10** or higher. We rely on some modern libraries, so keeping your environment up-to-date is recommended.

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/emonmorol/ANN-hybrid-nanofluid-model.git
   cd ANN-hybrid-nanofluid-model
   ```

2. **Install dependencies:**
   We've bundled everything you need in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

**Key Libraries:**
- **`torch`**: For building and training the neural network
- **`scipy`**: For the numerical ODE solver (`solve_bvp`)
- **`numpy` & `pandas`**: For data manipulation and analysis
- **`matplotlib` & `seaborn`**: For creating publication-quality plots
- **`joblib`**: For saving/loading scalers and models

---

## ⚡ Quick Start

Want to see it in action? Here's how to get started:

### The "Do Everything" Command
This will generate data, train the model, and save everything in one go:
```bash
python src/main.py all
```

### Step-by-Step Workflow
If you prefer to run things manually and understand each step:

1. **Generate the Training Data:**
   ```bash
   python src/main.py generate
   ```
   *This solves the ODEs for 81 parameter combinations, creating ~32,400 data points.*

2. **Train the Neural Network:**
   ```bash
   python src/main.py train
   ```
   *Watch the loss decrease as the model learns! 📉*

3. **Create Visualizations:**
   ```bash
   python src/visualizer.py
   ```
   *Generate beautiful plots comparing predictions with test data.*

4. **Validate the Model:**
   ```bash
   python src/validate_model.py
   ```
   *Check comprehensive error metrics and validation plots.*

---
