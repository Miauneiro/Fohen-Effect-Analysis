import matplotlib.pyplot as plt
import numpy as np
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc

import matplotlib.pyplot as plt
import numpy as np
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc


class FoehnEffectAnalyzer:
    """
    Analyzer for Föhn effect using thermodynamic diagrams.
    
    Attributes
    ----------
    skew : metpy.plots.SkewT
        Skew-T Log-P diagram object
    initial_conditions : dict
        Initial atmospheric conditions on windward side
    final_conditions : dict
        Final atmospheric conditions on leeward side
    """
    
    def __init__(self, fig_size=(12, 10)):
        """
        Initialize the Föhn effect analyzer.
        
        Parameters
        ----------
        fig_size : tuple, optional
            Figure size in inches (width, height)
        """
        self.fig = plt.figure(figsize=fig_size)
        self.skew = SkewT(self.fig, rotation=45)
        self.initial_conditions = {}
        self.final_conditions = {}
        
    def configure_diagram(self, p_min=350, p_max=1020, t_min=-10, t_max=40):
        """
        Configure the Skew-T diagram appearance and limits.
        
        Parameters
        ----------
        p_min : float
            Minimum pressure (hPa)
        p_max : float
            Maximum pressure (hPa)
        t_min : float
            Minimum temperature (°C)
        t_max : float
            Maximum temperature (°C)
        """
        # Add reference lines
        self.skew.plot_dry_adiabats(alpha=0.3, color='grey', linewidth=0.8)
        self.skew.plot_moist_adiabats(alpha=0.3, color='red', linewidth=0.8)
        self.skew.plot_mixing_lines(
            alpha=0.3, 
            color='tab:cyan', 
            linewidth=0.8,
            pressure=np.linspace(p_max, p_min, 1000) * units.hPa
        )
        
        # Configure axes
        self.skew.ax.set_ylim(p_max, p_min)
        self.skew.ax.set_xlim(t_min, t_max)
        self.skew.ax.set_xlabel('Temperature (°C)', fontsize=14)
        self.skew.ax.set_ylabel('Pressure (hPa)', fontsize=14)
        self.skew.ax.tick_params(axis='both', which='major', labelsize=12)
        
    def set_initial_conditions(self, pressure, temperature, dewpoint, mixing_ratio):
        """
        Set initial atmospheric conditions (windward side).
        
        Parameters
        ----------
        pressure : float
            Pressure in hPa
        temperature : float
            Temperature in °C
        dewpoint : float
            Dewpoint temperature in °C
        mixing_ratio : float
            Mixing ratio in g/kg
        """
        self.initial_conditions = {
            'pressure': pressure * units.hPa,
            'temperature': temperature * units.degC,
            'dewpoint': dewpoint * units.degC,
            'mixing_ratio': mixing_ratio * units('g/kg')
        }
        
    def set_final_conditions(self, pressure, temperature, dewpoint, mixing_ratio):
        """
        Set final atmospheric conditions (leeward side).
        
        Parameters
        ----------
        pressure : float
            Pressure in hPa
        temperature : float
            Temperature in °C
        dewpoint : float
            Dewpoint temperature in °C
        mixing_ratio : float
            Mixing ratio in g/kg
        """
        self.final_conditions = {
            'pressure': pressure * units.hPa,
            'temperature': temperature * units.degC,
            'dewpoint': dewpoint * units.degC,
            'mixing_ratio': mixing_ratio * units('g/kg')
        }
        
    def plot_observation_point(self, pressure, temperature, color, label, is_dewpoint=False):
        """
        Plot an observation point on the Skew-T diagram.
        
        Parameters
        ----------
        pressure : pint.Quantity
            Pressure level
        temperature : pint.Quantity
            Temperature
        color : str
            Marker face color
        label : str
            Label for legend
        is_dewpoint : bool, optional
            Whether this is a dewpoint observation
        """
        self.skew.plot(
            pressure, temperature, 'o',
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=2.5,
            markerfacecolor=color,
            label=label
        )
        
    def plot_adiabatic_process(self, p_start, p_end, t_start, process_type='dry', 
                               color='grey', linestyle='-', label=''):
        """
        Plot an adiabatic process on the Skew-T diagram.
        
        Parameters
        ----------
        p_start : float
            Starting pressure (hPa)
        p_end : float
            Ending pressure (hPa)
        t_start : pint.Quantity
            Starting temperature
        process_type : str
            Type of process ('dry' or 'moist')
        color : str
            Line color
        linestyle : str
            Line style
        label : str
            Label for legend
        """
        pressures = np.linspace(p_start, p_end, 50) * units.hPa
        
        if process_type == 'dry':
            temperatures = mpcalc.dry_lapse(pressures, t_start)
        elif process_type == 'moist':
            ref_pressure = pressures[0]
            temperatures = mpcalc.moist_lapse(pressures, t_start, ref_pressure)
        else:
            raise ValueError("process_type must be 'dry' or 'moist'")
            
        self.skew.plot(pressures, temperatures, color=color, 
                      linewidth=2.5, linestyle=linestyle, label=label)
        
    def plot_mixing_ratio_line(self, mixing_ratio, p_start, p_end, color='tab:cyan'):
        """
        Plot a constant mixing ratio line.
        
        Parameters
        ----------
        mixing_ratio : float
            Mixing ratio in g/kg
        p_start : float
            Starting pressure (hPa)
        p_end : float
            Ending pressure (hPa)
        color : str
            Line color
        """
        pressures = np.linspace(p_start, p_end, 50) * units.hPa
        self.skew.plot_mixing_lines(
            mixing_ratio=[mixing_ratio] * units('g/kg'),
            pressure=pressures,
            colors=color,
            linewidth=2.5,
            alpha=0.9
        )
        
    def calculate_relative_humidity(self, temperature, dewpoint):
        """
        Calculate relative humidity from temperature and dewpoint.
        
        Parameters
        ----------
        temperature : pint.Quantity
            Air temperature
        dewpoint : pint.Quantity
            Dewpoint temperature
            
        Returns
        -------
        pint.Quantity
            Relative humidity as percentage
        """
        return mpcalc.relative_humidity_from_dewpoint(temperature, dewpoint)
    
    def add_results_annotation(self, results_dict):
        """
        Add a text box with analysis results.
        
        Parameters
        ----------
        results_dict : dict
            Dictionary containing analysis results
        """
        text = self._format_results_text(results_dict)
        self.skew.ax.text(
            0.78, 0.98,
            text,
            transform=self.skew.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        )
        
    def _format_results_text(self, results):
        """Format results dictionary into annotation text."""
        return (
            f"Windward Side (North):\n"
            f"  T = {results['initial_temp']:.1f}°C, "
            f"Td = {results['initial_dewpoint']:.1f}°C\n"
            f"  r = {results['initial_mixing_ratio']:.1f} g/kg\n\n"
            f"LCL (ascent): {results['lcl_ascent_p']:.0f} hPa, "
            f"{results['lcl_ascent_t']:.1f}°C\n\n"
            f"Summit: {results['summit_p']:.0f} hPa, "
            f"{results['summit_t']:.1f}°C\n\n"
            f"LCL (descent): {results['lcl_descent_p']:.0f} hPa, "
            f"{results['lcl_descent_t']:.1f}°C\n\n"
            f"Leeward Side (South):\n"
            f"  T = {results['final_temp']:.1f}°C, "
            f"Td = {results['final_dewpoint']:.1f}°C\n"
            f"  r = {results['final_mixing_ratio']:.1f} g/kg\n"
            f"  RH = {results['final_rh']:.1f}%"
        )
        
    def finalize_plot(self, title, legend_loc='center left'):
        """
        Finalize the plot with title and legend.
        
        Parameters
        ----------
        title : str
            Plot title
        legend_loc : str
            Legend location
        """
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc=legend_loc, fontsize=9, ncol=1)
        plt.tight_layout()
        
    def save_figure(self, filename, dpi=300):
        """
        Save the figure to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        dpi : int
            Resolution in dots per inch
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        
    def show(self):
        """Display the plot."""
        plt.show()
