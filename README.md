# Föhn Effect Analysis - Madeira Island

This project conducts a meteorological analysis of the Föhn (or Foehn) effect on Madeira Island. Using data from the windward (North) and leeward (South) sides of the island, it leverages `metpy` and `matplotlib` to plot the complete thermodynamic path of an air parcel on a Skew-T Log-P diagram.

The analysis visualizes the orographic lifting, condensation (precipitation), and subsequent adiabatic warming and drying that leads to the characteristic warm, dry conditions in Funchal.

## The Analysis

[cite_start]The core analysis, found in `notebooks/fohen_effect_madeira.ipynb`, models the following thermodynamic process:

1.  **Windward Side (North Coast):** An air parcel starts at 1000 hPa with a temperature of 20°C and a dewpoint of 10.5°C.
2.  **Dry Ascent:** The parcel rises and cools at the dry adiabatic lapse rate (DALR) until it becomes saturated.
3.  **LCL (Cloud Base):** Saturation is reached at the Lifting Condensation Level (LCL) at 870 hPa and 8.4°C. Clouds form.
4.  **Moist Ascent:** The saturated parcel continues rising over the summit (400 hPa, -32°C), cooling at the moist adiabatic lapse rate (SALR) and losing moisture as precipitation.
5.  **Moist Descent:** The parcel descends from the summit to the leeward cloud base (655 hPa, -4.5°C) at the SALR.
6.  **Dry Descent (Föhn Effect):** Now dry, the parcel descends rapidly to the leeward side (Funchal), warming at the DALR.
7.  **Leeward Side (Funchal):** The parcel arrives at 1000 hPa, now significantly **warmer (30°C)** and **drier (16.0% RH)**, demonstrating the Föhn effect.

## Results

### Summary of Conditions

| Location | Pressure | Temp. | Dewpoint | Mixing Ratio | Rel. Humidity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Windward (North)** | 1000 hPa | 20.0°C | 10.5°C | 8.0 g/kg | - |
| **LCL (Ascent)** | 870 hPa | 8.4°C | 8.4°C | 8.0 g/kg | 100% |
| **Summit** | 400 hPa | -32.0°C | -32.0°C | - | 100% |
| **LCL (Descent)** | 655 hPa | -4.5°C | -4.5°C | 4.2 g/kg | 100% |
| **Leeward (Funchal)**| 1000 hPa | 30.0°C | 1.4°C | 4.2 g/kg | **16.0%** |

### Output: Skew-T Log-P Diagram

The analysis generates the following Skew-T diagram, plotting the full thermodynamic path.

![Skew-T Diagram of Föhn Effect in Madeira](outputs/image.png)

## Project Structure

fohen-effect-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── skewt_plot.py
│   └── thermodynamic_analysis.py
├── data/
│   └── madeira_observations.json
├── notebooks/
│   └── fohen_effect_madeira.ipynb
├── outputs/
│   └── skewt_diagram.png
└── docs/
    └── methodology.md

## Technology Used

* **Python**
* **MetPy:** For meteorological calculations (`mpcalc`) and Skew-T plots (`SkewT`).
* **Matplotlib:** For plotting.
* **NumPy:** For numerical operations.
* **Jupyter Notebook:** For the analysis notebook.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fohen-effect-analysis.git](https://github.com/your-username/fohen-effect-analysis.git)
    cd fohen-effect-analysis
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the analysis:**
    Open and run the Jupyter Notebook:
    ```bash
    jupyter notebook notebooks/fohen_effect_madeira.ipynb
    ```
