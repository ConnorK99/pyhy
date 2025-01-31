"""A GUI used to create the .inf files for Hyades simulations.

The GUI does **not** require a local installation of Hyades to write a .inf.
The `Run Hyades` button **does** require a local installation of Hyades to work.

Example:
    Start the GUI with the following line::
        $ python inf_GUI.py
"""
import os
import pathlib
import matplotlib
import numpy as np
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from tools.inf_GUI_helper import Layer, InfWriter, LayerTab
from tools import hyades_runner
matplotlib.use("TkAgg")


class InputGUI:
    """Create a GUI to format Hyades .inf files.

    Takes in many parameters, including simulation variables, Pressure / Temperature / Laser drives, and
    material properties then formats it all into a .inf for Hyades. Also has options to run Hyades and the optimizer.

    Note:
        Using a thermal model requires you have the Hyades formatted thermal conductivity table in the same directory
        as the .inf. Specifying the thermal model requires the filename in the .inf, so you will have to manually edit
        the .inf with the local filename.

        Specifying a constant thermal conductivity does not require any external tables.

        This GUI does not provide control over all possible Hyades inputs. See the Hyades Users' Guide for complete
        documentation.

        The `Write Inf` button will overwrite files with the same name without warning.
        Always double check your filenames.

    """
    def __init__(self, root):
        """Create GUI variables and format the entire thing. This generates the GUI window."""
        self.tabs = []
        self.n_layers = IntVar()
        self.time_max = DoubleVar()
        self.time_step = DoubleVar()
        self.pres_fname = StringVar()
        self.temp_fname = StringVar()
        self.laser_fname = StringVar()
        self.laser_wavelength = DoubleVar()
        self.laser_spot_diameter = DoubleVar()
        self.out_dir = StringVar()
        self.out_fname = StringVar()
        self.is_xray_probe = IntVar()  # binary 0/1 for False/True
        self.xray_probe_start = DoubleVar()
        self.xray_probe_stop = DoubleVar()
        self.source_multiplier = DoubleVar()
        self.time_of_interestS = DoubleVar(root)
        self.time_of_interestE = DoubleVar(root)
        self.exp_file_name = StringVar(root)
        self.save_excel = IntVar()
        self.save_excel.set(0)

        # Set up the window and root of all the widgets
        root.title('PyHy Input File GUI')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Parent most widgets will use
        self.parent = ttk.Frame(root, padding=' 3 3 12 12').grid(column=0, row=0, sticky='NW')

        row = 1
        # Add titles
        Label(self.parent, text='PyHy Input File GUI',
              font=('Arial', 14)).grid(column=1, row=row, columnspan=4, pady=(5, 0))
        row += 1
        ttk.Label(self.parent, text='★ are required',).grid(column=1, row=row, columnspan=4, pady=(0, 5))
        row += 1

        def simulate():
            """Function to run all .inf files in a directory for Run Hyades button"""
            inf_path = os.path.join('.', 'data', 'inf')  # './data/inf/'
            final_destination = os.path.join('.', 'data')  # './data/'
            if self.save_excel.get() == 1:
                excel_variables = ['Pres', 'U', 'Rho', 'Te']
            else:
                excel_variables = []

            title = 'Hyades Input File GUI'
            files = [f for f in os.listdir(inf_path) if f.endswith('.inf')]
            message = f'Do you want to start {len(files)} Hyades Simulations?'
            message += f'\nInput files in {inf_path}: {", ".join(files)}'
            if len(files) == 0:
                messagebox.showerror(title, f'Found no .inf files in {inf_path}')
            elif messagebox.askyesno(title, message):
                hyades_runner.batch_run_hyades(inf_path, final_destination, excel_variables=excel_variables)

        tv_input_dir = os.path.join('.', 'data', 'tvInputs')

        def select_pres_file():
            """Function to select the pressure drive file"""
            pres_filename = filedialog.askopenfilename(initialdir=tv_input_dir, title='Select Pressure Profile')
            self.pres_fname.set(pres_filename)
            self.pres_label_variable.set(os.path.basename(pres_filename))

        def select_temp_file():
            """Function to select the temperature drive file"""
            temp_filename = filedialog.askopenfilename(initialdir=tv_input_dir, title='Select Temperature Profile')
            self.temp_fname.set(temp_filename)
            self.temp_label_variable.set(os.path.basename(temp_filename))

        def select_laser_file():
            """Function to select the laser drive file"""
            laser_filename = filedialog.askopenfilename(initialdir=tv_input_dir, title='Select Laser Profile')
            self.laser_fname.set(laser_filename)
            self.laser_label_variable.set(os.path.basename(laser_filename))

        def select_dir():
            """Function to select the destination to write the .infs"""
            out_dir = filedialog.askdirectory(initialdir=os.path.join('.', 'data'), title='Select .inf destination')
            self.out_dir.set(out_dir)

        # out_fname entry
        pad_y = (0, 2)
        ttk.Label(self.parent, text='★Output inf Name ').grid(row=row, column=1,
                                                              sticky='NE', pady=pad_y)
        ttk.Entry(self.parent, textvariable=self.out_fname).grid(row=row, column=2, columnspan=2,
                                                                 sticky='NWE', pady=pad_y)
        row += 1
        # Optionally select directory for inf
        pad_y = (0, 0)
        self.out_dir.set(os.path.join('.', 'data', 'inf'))
        ttk.Button(root, text='Select .inf destination', command=select_dir).grid(row=row, column=2,
                                                                                  sticky='NWE', pady=pad_y)
        Label(root, textvariable=self.out_dir).grid(row=row, column=3, sticky='NW', pady=pad_y)
        row += 1

        # time_max and time_step entries
        ttk.Label(self.parent, text='★Simulation Time (ns) ').grid(row=row, column=1, sticky='NE')
        ttk.Entry(self.parent, textvariable=self.time_max, width=7).grid(row=row, column=2, sticky='NW')
        # Run hyades button
        ttk.Button(self.parent, text='Run Hyades', command=simulate).grid(row=row, column=3, sticky='NWE')
        # Checkbutton to save a copy of all the hyades data as an excel sheet. Default False.
        ttk.Checkbutton(self.parent, text="Save Excel copy",
                        variable=self.save_excel).grid(row=row, column=4, sticky="NW")
        row += 1

        # Post Processor time step
        ttk.Label(self.parent, text='★Time Step (ns) ').grid(row=row, column=1, sticky='NE')
        ttk.Entry(self.parent, textvariable=self.time_step, width=7).grid(row=row, column=2, sticky='NW')
        # Write the .inf button
        ttk.Button(self.parent, text='Write inf', command=self.write_out_props).grid(row=row, column=3, sticky='NWE')
        row += 1

        # Add the number of layers entry and button
        pad_y = (0, 0)
        ttk.Label(self.parent, text='★Number of Layers ').grid(column=1, row=row, sticky='NE', pady=pad_y)
        ttk.Entry(self.parent, textvariable=self.n_layers, width=7).grid(column=2, row=row, sticky='NW', pady=pad_y)
        ttk.Button(self.parent, text='Generate Layers',
                   command=self.generate_layers).grid(column=3, row=row, sticky='NWE', pady=pad_y)
        row += 10  # Buffer for the inputs to specify the Layer information

        # Optional X-ray probe time
        pad_y = (0, 0)
        Label(self.parent, text='X-Ray Start Time (ns) ').grid(row=row, column=1, sticky='NE', pady=pad_y)
        ttk.Entry(self.parent, textvariable=self.xray_probe_start, width=7).grid(row=row, column=2, sticky='NW')
        Label(self.parent, text='X-Ray Stop Time (ns) ').grid(row=row, column=3, sticky='NE', pady=pad_y)
        ttk.Entry(self.parent, textvariable=self.xray_probe_stop, width=7).grid(row=row, column=4, sticky='NW')
        row += 1

        # Select tv inputs from two column .txt files
        self.source_multiplier.set(1.0)
        Label(root, text='Source Multiplier ').grid(row=row, column=1, sticky='NE',)
        ttk.Entry(root, textvariable=self.source_multiplier, width=7).grid(row=row, column=2, sticky='NW')
        Label(root, text='Source Multiplier scales all inputs').grid(row=row, column=3, columnspan=2, sticky='NW')
        row += 1

        self.temp_label_variable = StringVar()
        self.temp_label_variable.set('None selected')
        Label(root, textvariable=self.temp_label_variable).grid(row=row, column=2, sticky='NW')
        ttk.Button(root, text='Select Temperature', command=select_temp_file).grid(row=row, column=1,
                                                                                   sticky='NWE', padx=(7, 0))
        row += 1

        self.pres_label_variable = StringVar()
        self.pres_label_variable.set('None selected')
        Label(root, textvariable=self.pres_label_variable).grid(row=row, column=2, sticky='NW')
        ttk.Button(root, text='Select Pressure', command=select_pres_file).grid(row=row, column=1,
                                                                                sticky='NWE', padx=(7, 0))
        self.is_optimize_pressure = IntVar()  # 0/1 for False/True
        ttk.Checkbutton(root, text='Set Pressure for Optimization',
                        variable=self.is_optimize_pressure).grid(column=3, columnspan=2, row=row, sticky='NW')
        row += 1

        self.laser_label_variable = StringVar()
        self.laser_label_variable.set('None selected')
        Label(root, textvariable=self.laser_label_variable).grid(row=row, column=2, sticky='NW')
        ttk.Button(root, text='Select Laser', command=select_laser_file).grid(row=row, column=1,
                                                                              sticky='NWE', padx=(7, 0))
        row += 1
        # Additional Laser Parameters - these are from experiment
        Label(root, text='Laser Wavelength (nm)').grid(row=row, column=1, sticky='NE',)
        ttk.Entry(root, textvariable=self.laser_wavelength, width=7).grid(row=row, column=2, sticky='NW')
        Label(root, text='Laser Spot Diameter (mm)').grid(row=row, column=3, sticky='NE')
        ttk.Entry(root, textvariable=self.laser_spot_diameter, width=7).grid(row=row, column=4, sticky='NW')
        row += 1

    def generate_layers(self):
        """Generate the layer options inside the GUI"""
        self.tabs = []  # reset layers so they do not keep appending
        notebook = ttk.Notebook(self.parent)  # reset the notebook
        notebook.grid(column=1, row=9, columnspan=9, sticky='NWE', padx=(7, 7))
        for i in range(self.n_layers.get()):
            frame = ttk.Frame(notebook)
            tab = LayerTab(frame)
            tab.add_props()
            self.tabs.append(tab)
            notebook.add(frame, text=f'Layer {i+1}')

    def display(self):
        """Debugging function to display all variables in all layers"""
        for i, T in enumerate(self.tabs):
            print(f'Layer {i}')
            for k in vars(T):
                if k == 'parent' or k == 'row':
                    continue
                else:
                    print(f'{k}: {vars(T)[k].get()}')
            print()

    def get_tv(self, fname, var):
        """Read two column csv for tv inputs (in SI units) and convert them to Hyades units (cgs).

        Note:
            Temperatures are converted from degrees Kelvin to kiloElectron Volts.
            Pressures are converted from gigapascals to dynes / cm^2
            Laser Power (Terawatts) is converted to an intensity (ergs / second) by dividing by the spot size
            and multiplying by the unit conversion to get terajoules to ergs

        """
        df = pd.read_csv(fname, skiprows=1)
        time_column = df.columns[0]
        var_column = df.columns[1]
        if 'te' in var.lower():
            scale = 1 / 11604000  # degrees kelvin to kiloelectron volts
        elif 'pres' in var.lower():
            scale = 1e10  # GPa to dynes / cm^2
        elif 'laser' in var.lower():
            laser_spot_diameter = self.laser_spot_diameter.get() / 10  # convert mm to cm
            spot_area = np.pi * (laser_spot_diameter / 2)**2  # pi * r^2
            scale = (1 / spot_area) * 1e19  # TeraWatts to ergs / sec
        else:
            raise Exception(f'Unknown variable requested from get_tv: {var}')

        tv_lines = []
        for t, v in zip(df[time_column], df[var_column]):
            line = f'tv {t * 1e-9:.2e} {v * scale:.2e}'
            tv_lines.append(line)

        return tv_lines

    def write_out_props(self):
        """Convert the GUI properties to a Layer object then pass all the Layers to the InfWriter"""

        '''The user may enter invalid options. These if statements check for invalid options and notify the user.'''
        messagebox_title = 'Hyades Input File GUI'
        if not self.out_fname.get():  # Notify user if they did not enter a filename for the inf
            messagebox.showerror(messagebox_title,
                                 'Enter a unique filename for the inf.'
                                 '\nInf will not be written.')
            return

        if self.time_max.get() <= 0:  # Notify user for invalid simulation time
            messagebox.showerror(messagebox_title,
                                 'Simulation Time (ns) must be greater than zero.'
                                 '\nInf will not be written.')
            return
        if self.time_step.get() <= 0:  # Notify user for invalid time step
            messagebox.showerror(messagebox_title,
                                 'Time Step (ns) must be greater than zero. 0.1 ns is standard.'
                                 '\nInf will not be written.')
            return
        if self.n_layers.get() <= 0:  # notify user if they did not enter number of layers
            messagebox.showerror(messagebox_title,
                                 'Enter the number of layers then click Generate Layers.'
                                 '\nInf will not be written')
            return
        if any([len(tab.material.get()) == 0 for tab in self.tabs]):  # notify user if they didn't select a material
            invalid_layers = [f'Layer {i+1}' for i, tab in enumerate(self.tabs) if len(tab.material.get()) == 0]
            messagebox.showerror(messagebox_title,
                                 f'Select a material for {", ".join(invalid_layers)}.'
                                 f'\nInf will not be written.')
            return
        if any([tab.thickness.get() <= 0 for tab in self.tabs]):  # Notify user for invalid thickness
            invalid_materials = [f'Layer {i+1}' for i, tab in enumerate(self.tabs) if tab.thickness.get() <= 0]
            # invalid_materials = [tab.material.get() for tab in self.tabs if tab.thickness.get() <= 0]
            messagebox.showerror('Hyades Input File GUI',
                                 f'Enter a thickness greater than 0 for {", ".join(invalid_materials)}'
                                 f'\nInf will not be written.')
            return
        if any([tab.n_mesh.get() <= 0 for tab in self.tabs]):  # Notify user for invalid mesh count
            invalid_materials = [f'Layer {i+1}' for i, tab in enumerate(self.tabs) if tab.n_mesh.get() <= 0]
            # invalid_materials = [tab.material.get() for tab in self.tabs if tab.n_mesh.get() <= 0]
            messagebox.showerror('Hyades Input File GUI',
                                 f'Enter a Num Mesh Points greater than 0 for {", ".join(invalid_materials)}'
                                 f'\nInf will not be written.')
            return
        out_fname_without_extension = os.path.splitext(self.out_fname.get())[0]
        inf_files_without_extensions = [os.path.splitext(f)[0] for f in os.listdir(os.path.join('data', 'inf'))]
        if out_fname_without_extension in os.listdir('data'):  # Notify user if inf name is already in pyhy/data
            messagebox.showerror(messagebox_title,
                                 f'{self.out_fname.get()!r} was previously used as an inf name. '
                                 f'Try using another name or remove the old folder in pyhy/data.'
                                 f'\nInf will not be written.')
            return
        if out_fname_without_extension in inf_files_without_extensions:  # Notify user if inf name already in data/inf
            response = messagebox.askyesno(messagebox_title,
                                           f'{self.out_fname.get()!r} already exists in pyhy/data/inf.'
                                           f'\nDo you want to replace the old inf with the current GUI settings?')
            if response:  # If user clicks yes
                messagebox.showinfo(messagebox_title,
                                    f'The current GUI settings will be written to {self.out_fname.get()}.inf.'
                                    f'\nThe old file with the same name will be overwritten.')
                # No return statement as this case continues onto further valid input checks
            else:  # If user clicks no
                messagebox.showinfo(messagebox_title,
                                    f'The current GUI settings were not written to an inf.'
                                    f'\nTry another name or remove the old file in pyhy/data/inf.')
                return  # exits write_out_props and does NOT write inf

        layers = []
        for i, T in enumerate(self.tabs):
            prop_dict = {}  # scraps all the properties out of GUI
            for prop in vars(T):
                if prop == 'parent' or prop == 'row':
                    continue
                else:
                    prop_dict[prop] = vars(T)[prop].get()
            i_layer = Layer(prop_dict)  # Layer class fills in missing layer info
            layers.append(i_layer)

        sim_props = {'time_max': self.time_max.get(),
                     'time_step': self.time_step.get(),
                     'sourceMultiplier': self.source_multiplier.get()
                     }

        if self.is_optimize_pressure.get() == 1:
            if not sum([L.is_material_of_interest for L in layers]) == 1:
                messagebox.showerror("Hyades Input File GUI",
                                     "Exactly one layer must be selected as Material of Interest when optimizing")
                raise Exception("Exactly one layer must be selected as Material of Interest when optimizing")
            if not sum([L.is_shock_material_of_interest for L in layers]) < 2:
                messagebox.showerror("Hyades Input File GUI",
                                     "No more than 1 layer can be selected as Shock MOI when optimizing")
                raise Exception("No more than 1 layer can be selected as Shock MOI when optimizing")
            sim_props['tvPres'] = ['TV_PRES']
        elif ('None' not in self.pres_fname.get()) and (self.pres_fname.get()):
            sim_props['tvPres'] = self.get_tv(self.pres_fname.get(), 'pres')

        if ('None' not in self.temp_fname.get()) and (self.temp_fname.get()):
            sim_props['tvTemp'] = self.get_tv(self.temp_fname.get(), 'temp')
        if ('None' not in self.laser_fname.get()) and (self.laser_fname.get()):
            sim_props['tvLaser'] = self.get_tv(self.laser_fname.get(), 'laser')
            sim_props['laserWavelength'] = self.laser_wavelength.get()
        if (self.xray_probe_stop.get() != 0) and (self.xray_probe_start.get() != 0):
            sim_props['xray_probe_start'] = self.xray_probe_start.get()
            sim_props['xray_probe_stop'] = self.xray_probe_stop.get()

        writer = InfWriter()
        writer.add_layers(layers, sim_props)  # put layers & simulation properties in the InfWriter
        # writer.display()  # displays a formatted inf file

        '''Last round of checks on user input'''
        if any([L.max_zone_width > 1 for L in layers]):  # Check for Zones greater than 1 micron created by increments
            wide_layers = [f'Layer{i + 1}' for i, L in enumerate(layers) if L.max_zone_width > 1]
            response = messagebox.askyesno(messagebox_title,
                                           f'{", ".join(wide_layers)} contain zones wider than 1 micron. '
                                           f'This is likely caused by low mesh resolution or extreme increments. '
                                           f'Are you sure you want to continue writing this inf?')
            if response:  # User clicks yes
                pass
            else:  # User clicks no
                return  # exits and does NOT write inf
        '''End checks on user input'''

        inf_dir = os.path.join('.', 'data', 'inf')
        if self.out_dir.get() == 'Select Directory':
            if os.path.isdir(inf_dir):
                out_dir = inf_dir  # './data/inf'
            else:
                out_dir = '.' + os.path.sep
        else:
            out_dir = self.out_dir.get()
        # If there isn't a directory for pyhy/data/inf, then make one
        if (out_dir == inf_dir) and (not os.path.exists(out_dir)):
            os.mkdir(out_dir)
        out_filename = os.path.join(out_dir, self.out_fname.get())
        writer.write_out(out_filename)


if __name__ == "__main__":
    root = Tk()
    style = ttk.Style(root)
    GUI = InputGUI(root)
    root.mainloop()
