import json
import os
from metpy.units import units
from .skewt_plot import FoehnEffectAnalyzer # Imports your class

# --- NEW: Define the path to the data file ---
# This builds a path from this file (src/ther...) up to the project root, 
# and then down into the data/ folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'madeira_observations.json')

# --- NEW: Helper function to load the data ---
def load_observation_data(data_file_path=DEFAULT_DATA_PATH):
    """Loads observation data from a JSON file."""
    with open(data_file_path, 'r') as f:
        return json.load(f)

# --- UPDATED: Main analysis function ---
def analyze_madeira_fohen_effect():
    """
    Main analysis function for Madeira Island Föhn effect.
    
    This function demonstrates the complete thermodynamic analysis of
    orographic lifting and the Föhn effect using observational data
    from Madeira Island.
    
    Returns
    -------
    FoehnEffectAnalyzer
        Analyzer object with complete analysis
    """
    
    # --- NEW: Load data from JSON instead of hard-coding ---
    data = load_observation_data()
    
    # Unpack data from the loaded dictionary
    north = data['windward']
    south = data['leeward']
    path = data['path_points']

    # Define atmospheric observations from the data
    # Windward side (North Coast)
    north_pressure = north['pressure']
    north_temp = north['temperature']
    north_dewpoint = north['dewpoint']
    north_mixing_ratio = north['mixing_ratio']
    
    # Leeward side (South Coast - Funchal)
    south_pressure = south['pressure']
    south_temp = south['temperature']
    south_dewpoint = south['dewpoint']
    south_mixing_ratio = south['mixing_ratio']
    
    # Lifting Condensation Level (ascent)
    lcl_ascent_pressure = path['lcl_ascent']['pressure']
    lcl_ascent_temp = path['lcl_ascent']['temperature']
    
    # Summit
    summit_pressure = path['summit']['pressure']
    summit_temp = path['summit']['temperature']
    
    # Lifting Condensation Level (descent)
    lcl_descent_pressure = path['lcl_descent']['pressure']
    lcl_descent_temp = path['lcl_descent']['temperature']
    
    # --- The rest of the function is the same as before ---
    
    # Initialize analyzer
    analyzer = FoehnEffectAnalyzer(fig_size=(12, 10))
    analyzer.configure_diagram()
    
    # Set conditions
    analyzer.set_initial_conditions(north_pressure, north_temp, 
                                    north_dewpoint, north_mixing_ratio)
    analyzer.set_final_conditions(south_pressure, south_temp, 
                                  south_dewpoint, south_mixing_ratio)
    
    # Plot observation points
    # North coast
    analyzer.plot_observation_point(
        analyzer.initial_conditions['pressure'],
        analyzer.initial_conditions['temperature'],
        'orange',
        f'Initial T ({north_temp}°C, {north_pressure} hPa)'
    )
    analyzer.plot_observation_point(
        analyzer.initial_conditions['pressure'],
        analyzer.initial_conditions['dewpoint'],
        'tab:blue',
        f'Initial Td ({north_dewpoint}°C)',
        is_dewpoint=True
    )
    
    # LCL (ascent)
    analyzer.plot_observation_point(
        lcl_ascent_pressure * units.hPa,
        lcl_ascent_temp * units.degC,
        'orange',
        f'LCL - Cloud Base ({lcl_ascent_pressure} hPa, {lcl_ascent_temp}°C)'
    )
    
    # Summit
    analyzer.plot_observation_point(
        summit_pressure * units.hPa,
        summit_temp * units.degC,
        'orange',
        f'Summit ({summit_pressure} hPa, {summit_temp}°C)'
    )
    
    # LCL (descent)
    analyzer.plot_observation_point(
        lcl_descent_pressure * units.hPa,
        lcl_descent_temp * units.degC,
        'orange',
        f'Cloud Base - Descent ({lcl_descent_pressure} hPa, {lcl_descent_temp}°C)'
    )
    
    # South coast (Funchal)
    analyzer.plot_observation_point(
        analyzer.final_conditions['pressure'],
        analyzer.final_conditions['temperature'],
        'orange',
        f'Final T - Funchal ({south_temp}°C)'
    )
    analyzer.plot_observation_point(
        analyzer.final_conditions['pressure'],
        analyzer.final_conditions['dewpoint'],
        'tab:blue',
        f'Final Td - Funchal ({south_dewpoint}°C)',
        is_dewpoint=True
    )
    
    # Plot adiabatic processes
    # 1. Dry adiabatic ascent (north coast to LCL)
    analyzer.plot_adiabatic_process(
        north_pressure, lcl_ascent_pressure,
        analyzer.initial_conditions['temperature'],
        process_type='dry',
        color='grey',
        linestyle='-',
        label='Dry Adiabatic Ascent'
    )
    
    # 2. Moist adiabatic ascent (LCL to summit)
    analyzer.plot_adiabatic_process(
        lcl_ascent_pressure, summit_pressure,
        lcl_ascent_temp * units.degC,
        process_type='moist',
        color='red',
        linestyle='-',
        label='Saturated Adiabatic Ascent'
    )
    
    # 3. Moist adiabatic descent (summit to LCL descent)
    analyzer.plot_adiabatic_process(
        summit_pressure, lcl_descent_pressure,
        summit_temp * units.degC,
        process_type='moist',
        color='red',
        linestyle='--',
        label='Saturated Adiabatic Descent'
    )
    
    # 4. Dry adiabatic descent (LCL to Funchal) - Föhn warming
    analyzer.plot_adiabatic_process(
        lcl_descent_pressure, south_pressure,
        lcl_descent_temp * units.degC,
        process_type='dry',
        color='grey',
        linestyle='--',
        label='Dry Adiabatic Descent (Föhn)'
    )
    
    # Plot mixing ratio lines
    analyzer.plot_mixing_ratio_line(north_mixing_ratio, north_pressure, 
                                    lcl_ascent_pressure)
    analyzer.plot_mixing_ratio_line(south_mixing_ratio, south_pressure, 
                                    lcl_descent_pressure)
    
    # Calculate final relative humidity
    final_rh = analyzer.calculate_relative_humidity(
        analyzer.final_conditions['temperature'],
        analyzer.final_conditions['dewpoint']
    )
    
    # Prepare results dictionary
    results = {
        'initial_temp': north_temp,
        'initial_dewpoint': north_dewpoint,
        'initial_mixing_ratio': north_mixing_ratio,
        'lcl_ascent_p': lcl_ascent_pressure,
        'lcl_ascent_t': lcl_ascent_temp,
        'summit_p': summit_pressure,
        'summit_t': summit_temp,
        'lcl_descent_p': lcl_descent_pressure,
        'lcl_descent_t': lcl_descent_temp,
        'final_temp': south_temp,
        'final_dewpoint': south_dewpoint,
        'final_mixing_ratio': south_mixing_ratio,
        'final_rh': final_rh.to('percent').m
    }
    
    # Print results
    print("\n" + "="*50)
    print("FÖHN EFFECT ANALYSIS - MADEIRA ISLAND")
    print("="*50)
    print(f"\nWindward Side (North Coast):")
    print(f"  Temperature: {north_temp}°C")
    print(f"  Dewpoint: {north_dewpoint}°C")
    print(f"  Mixing Ratio: {north_mixing_ratio} g/kg")
    print(f"\nOrographic Lifting:")
    print(f"  LCL (Cloud Base): {lcl_ascent_pressure} hPa, {lcl_ascent_temp}°C")
    print(f"  Summit: {summit_pressure} hPa, {summit_temp}°C")
    print(f"  LCL (Descent): {lcl_descent_pressure} hPa, {lcl_descent_temp}°C")
    print(f"\nLeeward Side (Funchal - South Coast):")
    print(f"  Temperature: {south_temp}°C")
    print(f"  Dewpoint: {south_dewpoint}°C")
    print(f"  Mixing Ratio: {south_mixing_ratio} g/kg")
    print(f"  Relative Humidity: {final_rh.to('percent').m:.1f}%")
    print(f"\nFöhn Effect:")
    print(f"  Temperature increase: {south_temp - north_temp}°C")
    print(f"  Moisture loss: {north_mixing_ratio - south_mixing_ratio} g/kg")
    print(f"  Result: Warm, dry conditions (RH < 20%)")
    print("="*50 + "\n")
    
    # Add annotation
    analyzer.add_results_annotation(results)
    
    # Finalize plot
    analyzer.finalize_plot('Skew-T Log-P Diagram - Föhn Effect in Madeira Island')
    
    return analyzer
