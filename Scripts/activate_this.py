import math
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List
import itertools
import  re
import copy
import numpy as np
import matplotlib.pyplot as plt
import cmath



class FileHandling:
    def __init__(self, path):
        self.path = path

    def read_signal_from_file(self):
        with open(self.path, 'r') as file:
            lines = file.readlines()

            signal_type = int(lines[0].strip())
            is_periodic = int(lines[1].strip())
            N1 = int(lines[2].strip())

            X = []
            Y = []
            for i in range(3, len(lines)):
                parts = re.split(r"[, \s]+", lines[i].strip())  # Split on comma or whitespace
                #print(parts)
                # Remove 'f' and convert to floats if both parts are found
                if len(parts) == 2:
                    x = float(parts[0].replace('f', ''))
                    y = float(parts[1].replace('f', ''))
                    #print(f"x: {x}, y: {y}")
                else:
                    print(f"Line does not contain two valid values")
                X.append(float(x))
                Y.append(float(y))

            signal_tmp = Signal(X, Y)
            return signal_tmp


class Signal:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
    def clear(self):
        self.X = []
        self.Y = []

class Task_1:
    def __init__(self, root):
        self.root = root
        self.create_widgets()
        self.stack: List[Signal([],[])] = []  # stack of signals
        self.last_signal = Signal([], [])
        self.current_mode = "signal"
        self.output_id = 0
        self.RGB = ['red','green','blue','yellow']
        self.rgb_idx = 0
        self.first_der = Signal([],[])
        self.second_der = Signal([],[])
        self.is_derevative = 0

    def QuantizationTest2(self,file_name,Your_IntervalIndices,Your_EncodedValues,Your_QuantizedValues,Your_SampledError):
        expectedIntervalIndices=[]
        expectedEncodedValues=[]
        expectedQuantizedValues=[]
        expectedSampledError=[]
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                if len(L.split(' '))==4:
                    L=line.split(' ')
                    V1=int(L[0])
                    V2=str(L[1])
                    V3=float(L[2])
                    V4=float(L[3])
                    expectedIntervalIndices.append(V1)
                    expectedEncodedValues.append(V2)
                    expectedQuantizedValues.append(V3)
                    expectedSampledError.append(V4)
                    line = f.readline()
                else:
                    break
        if(len(Your_IntervalIndices)!=len(expectedIntervalIndices)
            or len(Your_EncodedValues)!=len(expectedEncodedValues)
            or len(Your_QuantizedValues)!=len(expectedQuantizedValues)
            or len(Your_SampledError)!=len(expectedSampledError)):
            print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_IntervalIndices)):
            if(Your_IntervalIndices[i]!=expectedIntervalIndices[i]):
                print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one")
                return

        # for i in range(len(Your_EncodedValues)):
        #     print(Your_EncodedValues[i],expectedEncodedValues[i])

        for i in range(len(Your_EncodedValues)):
            if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
                print("QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")
                return

        for i in range(len(expectedQuantizedValues)):
            if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
                continue
            else:
                print("QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one")
                return
        for i in range(len(expectedSampledError)):
            print(Your_SampledError[i] ,expectedSampledError[i] )
        for i in range(len(expectedSampledError)):
            if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
                continue
            else:
                print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
                return

        print('QuantizationTest2 Test case passed successfully')

    def QuantizationTest1(self, file_name,Your_EncodedValues,Your_QuantizedValues):
        expectedEncodedValues=[]
        expectedQuantizedValues=[]
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                if len(L.split(' '))==2:
                    L=line.split(' ')
                    V2=str(L[0])
                    V3=float(L[1])
                    expectedEncodedValues.append(V2)
                    expectedQuantizedValues.append(V3)
                    line = f.readline()
                else:
                    break
        if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
            print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
            return

        for i in range(len(Your_EncodedValues)):
            print(Your_EncodedValues[i] , " ", expectedEncodedValues[i])

        for i in range(len(Your_EncodedValues)):
            if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
                print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
                return

        for i in range(len(expectedQuantizedValues)):
            print(Your_QuantizedValues[i]," - ",expectedQuantizedValues[i] )

        for i in range(len(expectedQuantizedValues)):
            if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
                continue
            else:
                print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
                return
        print("QuantizationTest1 Test case passed successfully")

    def SignalSamplesAreEqual(self, file_name, indices, samples):
        print(file_name)

        expected_indices = []
        expected_samples = []
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L = line.strip()
                if len(L.split(' ')) == 2:
                    L = line.split(' ')
                    V1 = int(L[0])
                    V2 = float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break
        #print(len(expected_samples) )
        #print(len(samples) )
        if len(expected_samples) != len(samples):
            print("Test case failed, your signal have different length from the expected one")
            return

        for i in range(len(expected_indices)):
            if (indices[i] != expected_indices[i]):

                print("Test case failed, your signal have different samples from the expected one")
                return

        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal have different values from the expected one")
                return
        print("Test case passed successfully")

    def Load_file(self):
        path = self.navigate_file()
        file = FileHandling(path)
        self.stack.append(file.read_signal_from_file())

    def clear_signal_stack(self):
        if len(self.stack) == 0:
            messagebox.showinfo("All are clean", "Your stack is empty!")
            return
        self.stack.pop()
        messagebox.showinfo("Operation", "The top signal was deleted!")


    def create_widgets(self):
        self.root.configure(bg='lightblue')

        self.load_button = tk.Button(self.root, text="Load Signal File", command=self.Load_file, bg='skyblue',fg='black')
        self.clear_button = tk.Button(self.root, text="Clear Last Signal", command=self.clear_signal_stack, bg='skyblue',fg='black')

        self.signal_type_label = tk.Label(self.root, text="Signal Type:", bg='lightblue', fg='darkblue')

        self.signal_type_combobox = ttk.Combobox(self.root, values=['sin', 'cos'])
        self.signal_type_combobox.set('sin')

        self.amplitude_label = tk.Label(self.root, text="Amplitude:", bg='lightblue', fg='darkblue')
        self.amplitude_entry = tk.Entry(self.root)
        self.amplitude_entry.insert(0,"0.0")

        self.analog_freq_label = tk.Label(self.root, text="Analog Frequency:", bg='lightblue', fg='darkblue')
        self.analog_freq_entry = tk.Entry(self.root)
        self.analog_freq_entry.insert(0,"0.0")

        self.sampling_freq_label = tk.Label(self.root, text="Sampling Frequency:", bg='lightblue', fg='darkblue')
        self.sampling_freq_entry = tk.Entry(self.root)
        self.sampling_freq_entry.insert(0,"0.0")

        self.phase_label = tk.Label(self.root, text="Phase Shift:", bg='lightblue', fg='darkblue')
        self.phase_entry = tk.Entry(self.root)
        self.phase_entry.insert(0,"0.0")

        self.min_value_norm_abel = tk.Label(self.root, text="min value:", bg='lightblue', fg='darkblue')
        self.min_value_norm_Entry = tk.Entry(self.root)
        self.min_value_norm_Entry.insert(0,"0.0")

        self.max_value_norm_abel = tk.Label(self.root, text="max value:", bg='lightblue', fg='darkblue')
        self.max_value_norm_Entry = tk.Entry(self.root)
        self.max_value_norm_Entry.insert(0,"0.0")

        self.signal_processing_options_label = tk.Label(self.root, text="Signal processing options:", bg='lightblue',
                                                 fg='darkblue')
        self.signal_processing_options_combobox = ttk.Combobox(self.root,
                                                              values=['add', 'sub', 'multiply', 'square', 'accumulate',
                                                                      'normalize','DFT','IDFT','DCT','Move Avg','remove DC','convolution',
                                                                      'norm cross correlation'])

        # Time domain parts
        self.time_domain_options_label = tk.Label(self.root, text="Signal processing options:", bg='lightblue',
                                                 fg='darkblue')
        self.time_domain_options_combobox = ttk.Combobox(self.root,
                                                              values=['sharp','shift','fold','fold and shift','remove DC',
                                                                     'filter','resample'])
        self.time_domain_options_combobox.bind("<<ComboboxSelected>>", self.on_time_domain_change)
        self.time_domain_options_combobox.set('sharp')
        self.shift_entry = tk.Entry(self.root)
        self.shift_entry.insert(0,"0")

        #filter stuff
        self.time_domain_filter_combobox = ttk.Combobox(self.root,
                                                              values=['Low pass','High pass','Band pass','Band stop'
                                                                     ])
        self.time_domain_filter_combobox.set('low pass')
        self.load_spec_butt = tk.Button(self.root, text="Load Spec", command=self.Load_spec_file, bg='skyblue',fg='black')

        self.stop_band_att_label = tk.Label(self.root, text="stop band att:", bg='lightblue', fg='darkblue')
        self.stop_band_att_entry = tk.Entry(self.root)
        self.stop_band_att_entry.insert(0,"0")

        self.FC1_label = tk.Label(self.root, text="FC_1:", bg='lightblue', fg='darkblue')
        self.FC1_entry = tk.Entry(self.root)
        self.FC1_entry.insert(0,"0")

        self.FC2_label = tk.Label(self.root, text="FC_2:", bg='lightblue', fg='darkblue')
        self.FC2_entry = tk.Entry(self.root)
        self.FC2_entry.insert(0,"0")

        self.trans_band_label = tk.Label(self.root, text="trans_band:", bg='lightblue', fg='darkblue')
        self.trans_band_entry = tk.Entry(self.root)
        self.trans_band_entry.insert(0,"0")

        # resampling stuff
        self.interpolate_L_label = tk.Label(self.root, text="Interpolation factor L:", bg='lightblue', fg='darkblue')
        self.interpolate_L_entry = tk.Entry(self.root)
        self.interpolate_L_entry.insert(0,"0")

        self.decimation_M_label = tk.Label(self.root, text="Decimation factor M:", bg='lightblue', fg='darkblue')
        self.decimation_M_entry = tk.Entry(self.root)
        self.decimation_M_entry.insert(0,"0")

        ############################
        self.signal_processing_options_combobox.bind("<<ComboboxSelected>>", self.on_processing_options_change)

        self.signal_processing_options_combobox.set('square')

        self.multiplication_entry = tk.Entry(self.root)
        self.multiplication_entry.insert(0, "0.0")

        self.FreqDFT_entry = tk.Entry(self.root)
        self.FreqDFT_entry.insert(0, "0.0")
        self.save_in_file_butt = tk.Button(self.root, text="save in file", command=self.save_in_file, bg='skyblue',
                                         fg='black')
        self.saved_signal_len_entry = tk.Entry(self.root)
        self.saved_signal_len_entry.insert(0,"0")

        self.signal_to_draw_label = tk.Label(self.root, text="Signal to draw:", bg='lightblue', fg='darkblue')
        self.signal_to_draw_combobox = ttk.Combobox(self.root, values=['signal', 'sincos signal', 'processed signal','Quantize','Time Domain'])
        self.signal_to_draw_combobox.bind("<<ComboboxSelected>>", self.on_to_draw_combo_change)

        self.signal_to_draw_combobox.set('signal')
        self.generate_button = tk.Button(self.root, text="Generate Signal", command=self.generate_signal, bg='skyblue',
                                         fg='black')

        self.compare_button = tk.Button(self.root, text="compare Signal Files", command=self.compare_signals,
                                        bg='skyblue', fg='black')

        self.signal_to_quantize = tk.Label(self.root, text="Signal quantize:", bg='lightblue', fg='darkblue')
        self.signal_to_quan_combobox = ttk.Combobox(self.root, values=['bit', 'level'])
        self.signal_to_quan_combobox.set('bit')
        self.quantize_entry = tk.Entry(self.root)
        self.quantize_entry.insert(0, "0")

        self.show_single_mode()

    def insert_entry(self,obj,value):
        obj.delete(0, tk.END)
        obj.insert(0, value)

    def pack_forget_all_time_domain(self):
        self.shift_entry.pack_forget()
        self.time_domain_filter_combobox.pack_forget()
        self.sampling_freq_label.pack_forget()
        self.sampling_freq_entry.pack_forget()
        self.stop_band_att_label.pack_forget()
        self.sampling_freq_entry.pack_forget()
        self.FC1_label.pack_forget()
        self.FC1_entry.pack_forget()
        self.FC2_label.pack_forget()
        self.FC2_entry.pack_forget()
        self.trans_band_label.pack_forget()
        self.trans_band_entry.pack_forget()
        self.load_spec_butt.pack_forget()
        self.stop_band_att_entry.pack_forget()
        self.decimation_M_label.pack_forget()
        self.decimation_M_entry.pack_forget()
        self.interpolate_L_label.pack_forget()
        self.interpolate_L_entry.pack_forget()
    def read_filter_specifications_from_file(self,filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        filter_specs = {}
        for line in lines:
            key, value = line.strip().split('=')
            filter_specs[key.strip()] = value.strip()

        return filter_specs
    def Load_spec_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            filter_specs = self.read_filter_specifications_from_file(file_path)
            self.time_domain_filter_combobox.set(filter_specs['FilterType'])
            self.insert_entry(self.sampling_freq_entry,(filter_specs['FS']))
            self.insert_entry(self.stop_band_att_entry,(filter_specs['StopBandAttenuation']))
            self.insert_entry(self.trans_band_entry,(filter_specs['TransitionBand']))

            # Update for Band pass and Band stop cases
            if filter_specs['FilterType'] in ['Band pass', 'Band stop']:
                self.insert_entry(self.FC2_entry,(filter_specs['F2']))
                self.insert_entry(self.FC1_entry,(filter_specs['F1']))

            else:
                self.insert_entry(self.FC1_entry,(filter_specs['FC']))
    def on_time_domain_change(self,event):
        selected_value = self.time_domain_options_combobox.get()
        print(selected_value)
        self.pack_forget_all_time_domain()
        if selected_value == 'shift' or selected_value == 'fold and shift':
            self.shift_entry.pack(pady=5)
        else:
            #self.shift_entry.pack_forget()
            if selected_value=='filter' or selected_value =='resample':
                self.time_domain_filter_combobox.pack(pady=1)
                self.load_spec_butt.pack(pady=1)
                self.sampling_freq_label.pack(pady=1)
                self.sampling_freq_entry.pack(pady=1)
                self.stop_band_att_label.pack(pady=1)
                self.stop_band_att_entry.pack(pady=1)
                self.sampling_freq_entry.pack(pady=1)
                self.FC1_label.pack(pady=1)
                self.FC1_entry.pack(pady=1)
                self.FC2_label.pack(pady=1)
                self.FC2_entry.pack(pady=1)
                self.trans_band_label.pack(pady=1)
                self.trans_band_entry.pack(pady=1)
                if selected_value == 'resample':
                    self.decimation_M_label.pack(pady=1)
                    self.decimation_M_entry.pack(pady=1)
                    self.interpolate_L_label.pack(pady=1)
                    self.interpolate_L_entry.pack(pady=1)
            else:
                messagebox.showinfo("ERROR","invalid choice in time domain combo box!")






    def save_in_file(self):
        self.output_id = self.output_id + 1
        N = int(self.saved_signal_len_entry.get())
        X = self.last_signal.X
        Y = self.last_signal.Y

        if N > len(X):
            messagebox.showinfo("Error", "Your desired size is not available in output signal!")
            return

        # Ensure the directory exists
        directory = os.path.join("saved_files")  # Use os.path.join for platform compatibility
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        file_path = os.path.join(directory, f"output_{self.output_id}.txt")  # Using f-string for path construction

        with open(file_path, 'w') as file:
            for i in range(N):
                file.write(f"{X[i]}  {Y[i]}\n")  # Add a newline for proper formatting

        print(os.path.abspath(file_path))


    def on_processing_options_change(self,event):
        # Get the selected value
        selected_value = self.signal_processing_options_combobox.get()
        self.min_value_norm_abel.pack_forget()
        self.min_value_norm_Entry.pack_forget()
        self.max_value_norm_abel.pack_forget()
        self.max_value_norm_Entry.pack_forget()
        self.multiplication_entry.pack_forget()
        self.FreqDFT_entry.pack_forget()

        # ['add', 'sub', 'multiply', 'square', 'accumulate', 'normalize','DCT']
        if selected_value == 'multiply' or selected_value == 'Move Avg':
            self.multiplication_entry.pack(pady=2)
        else:
            if selected_value == 'normalize':
                self.min_value_norm_abel.pack(pady=5)
                self.min_value_norm_Entry.pack(pady=2)
                self.max_value_norm_abel.pack(pady=5)
                self.max_value_norm_Entry.pack(pady=2)
            else:

                if selected_value =='DFT' or selected_value =='remove DC':
                    self.FreqDFT_entry.pack(pady=5)


    def hide_all_mode(self):
        self.load_button.pack_forget()
        self.generate_button.pack_forget()
        self.compare_button.pack_forget()
        self.clear_button.pack_forget()
        self.signal_to_draw_combobox.pack_forget()
        self.signal_to_draw_label.pack_forget()
        self.max_value_norm_abel.pack_forget()
        self.max_value_norm_Entry.pack_forget()
        self.min_value_norm_Entry.pack_forget()
        self.min_value_norm_abel.pack_forget()
        self.saved_signal_len_entry.pack_forget()
        self.save_in_file_butt.pack_forget()
        self.shift_entry.pack_forget()
        self.time_domain_options_combobox.pack_forget()

        self.signal_processing_options_label.pack_forget()
        self.signal_processing_options_combobox.pack_forget()
        self.multiplication_entry.pack_forget()
        self.signal_type_label.pack_forget()
        self.signal_type_combobox.pack_forget()
        self.amplitude_label.pack_forget()
        self.amplitude_entry.pack_forget()
        self.analog_freq_label.pack_forget()
        self.analog_freq_entry.pack_forget()
        self.sampling_freq_label.pack_forget()
        self.sampling_freq_entry.pack_forget()
        self.phase_label.pack_forget()
        self.phase_entry.pack_forget()
        self.quantize_entry.pack_forget()
        self.signal_to_quantize.pack_forget()
        self.signal_to_quan_combobox.pack_forget()
        self.FreqDFT_entry.pack_forget()

    def signal_mode(self):
        self.hide_all_mode()
        self.show_single_mode()



    def show_single_mode(self):
        self.load_button.pack(pady=5)
        self.clear_button.pack(pady=5)
        self.generate_button.pack(pady=5)
        self.signal_to_draw_label.pack(pady=5)
        self.signal_to_draw_combobox.pack(pady=2)


    def sincos_mode(self):
        self.hide_all_mode()
        self.show_sincos_mode()

    def show_sincos_mode(self):
        self.signal_type_label.pack(pady=5)
        self.signal_type_combobox.pack(pady=5)
        self.amplitude_label.pack(pady=5)
        self.amplitude_entry.pack(pady=5)
        self.analog_freq_label.pack(pady=5)
        self.analog_freq_entry.pack(pady=5)
        self.sampling_freq_label.pack(pady=5)
        self.sampling_freq_entry.pack(pady=5)
        self.phase_label.pack(pady=5)
        self.phase_entry.pack(pady=5)
        self.generate_button.pack(pady=5)
        self.compare_button.pack(pady=5)
        self.signal_to_draw_label.pack(pady=5)
        self.signal_to_draw_combobox.pack(pady=5)

    def show_process_signal_mode(self):

        self.compare_button.pack(pady=5)

        self.load_button.pack(pady=5)
        self.clear_button.pack(pady=5)
        self.generate_button.pack(pady=5)
        self.saved_signal_len_entry.pack(pady=5)
        self.save_in_file_butt.pack(pady=5)
        self.signal_to_draw_label.pack(pady=5)
        self.signal_to_draw_combobox.pack(pady=5)
        self.signal_processing_options_label.pack(pady=5)
        self.signal_processing_options_combobox.pack(pady=5)

    def Time_Domain_Mode(self):
        self.hide_all_mode()
        self.show_time_domain_mode()

    def show_time_domain_mode(self):
        self.compare_button.pack(pady=5)

        self.load_button.pack(pady=5)
        self.clear_button.pack(pady=5)
        self.generate_button.pack(pady=5)
        self.saved_signal_len_entry.pack(pady=5)
        self.save_in_file_butt.pack(pady=5)
        self.signal_to_draw_label.pack(pady=5)
        self.signal_to_draw_combobox.pack(pady=5)
        self.time_domain_options_label.pack(pady=5)
        self.time_domain_options_combobox.pack(pady=5)

    def process_signal_mode(self):
        self.hide_all_mode()
        self.show_process_signal_mode()

    def show_quantize_mode(self):
        self.compare_button.pack(pady=5)

        self.load_button.pack(pady=5)
        self.clear_button.pack(pady=5)
        self.generate_button.pack(pady=5)
        self.signal_to_quantize.pack(pady=5)
        self.signal_to_quan_combobox.pack(pady=5)
        self.quantize_entry.pack(pady=5)
        self.signal_to_draw_combobox.pack(pady=5)
    def quantize_mode(self):
        self.hide_all_mode()
        self.show_quantize_mode()

    def on_to_draw_combo_change(self,event):
        # Get the selected value
        selected_value = self.signal_to_draw_combobox.get()
        # ['signal', 'sincos signal', 'processed signal','Quantize']
        if selected_value == 'signal':
            self.signal_mode()
            self.current_mode= 'signal'
        elif selected_value == 'sincos signal':
            self.sincos_mode()
            self.current_mode = 'sincos'
        elif selected_value == 'processed signal':
            self.process_signal_mode()
            self.current_mode = 'process'
        elif selected_value == 'Quantize':
            self.quantize_mode()
            self.current_mode = 'Quantize'
        elif selected_value == 'Time Domain':
            self.Time_Domain_Mode()
            self.current_mode = 'Time Domain'
        else:
            print("error")



    def navigate_file(self):
        path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text files", "*.txt")])
        if path:
            print("Signal loaded successfully.")
            return path
        return None

    def single_signal_generator(self):
        if len(self.stack)==0:
            messagebox.showinfo("Error","There is no signal to draw in the system!")
            return

        sig = self.stack[len(self.stack)-1]
        self.plot_signals(sig.X, sig.Y, 'source signal')
        self.last_signal = sig

    def sincos_signal_generator(self):
        type = self.signal_type_combobox.get()
        amp = float(self.amplitude_entry.get())
        analogF = float(self.analog_freq_entry.get())
        samplingF = float(self.sampling_freq_entry.get())
        phase = float(self.phase_entry.get())

        # check if valid signal
        if samplingF < 2 * analogF:
            messagebox.showinfo("Error", "Your data may elise!")
            return

        n = np.linspace(0, samplingF, int(samplingF), endpoint=False)

        if type == 'sin':
            signal = amp * np.sin(2 * np.pi * analogF * n / samplingF + phase)
        elif type == 'cos':
            signal = amp * np.cos(2 * np.pi * analogF * n / samplingF + phase)
        else:
            messagebox.showinfo("Error","Invalid signal type. Choose 'sin' or 'cos'.")
            return

        self.last_signal = Signal(n.tolist(),signal)

        self.plot_signals(n, signal, 'sin_cos')



    def process_signal_generator(self):
        option = self.signal_processing_options_combobox.get()
        if option == 'add':
            self.add_two_signals(1)
        elif option == 'sub':
            self.add_two_signals(-1)
        elif option == 'multiply':
            self.multiply_signal_by_factor()
        elif option == 'square':
            self.square_signal()
        elif option ==  'accumulate':
            self.accumulate_signal()
        elif option == 'normalize':
            self.normalize_signal()
        elif option == 'DFT':
            self.DFT_signal()
        elif option == 'IDFT':
            self.DFT_signal(0,1)
        elif option == 'DCT':
            self.DCT_signal()
        elif option =='Move Avg':
            self.move_avg_signal()
        elif option == 'remove DC':
            self.remove_DC_freq()
        elif option == 'convolution':
            self.convlute_signal()
        elif option == 'norm cross correlation':
            self.norm_cross_correlation_signal()
        else:
            messagebox.showinfo("Option Error", "The seleceted option is undefined!")


    def compute_cross_correlation(self,x1, x2):
        N = len(x1)
        results = []

        for n in range(N):
            sum = 0
            for j in range(N):
                sum += x1[j]*x2[(j+n) % N]
            results.append(((1/N)*sum))

        return results

    def compute_normalize_correlation(self,x1,x2,corr):
        N = len(x1)
        results = []
        for n in range(N):
            n1_sum = 0
            n2_sum = 0
            for j in range(N):
                n1_sum += x1[j]**2
                n2_sum += x2[j]**2
            results.append(corr[n] / ((1/N)*math.sqrt(n1_sum*n2_sum)))

        return results
    def norm_cross_correlation_signal(self):
        if len(self.stack) < 2:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig2 = copy.deepcopy(self.stack[len(self.stack)-1])
        sig1 = copy.deepcopy(self.stack[len(self.stack)-2])

        len1 = len(sig1.Y)
        len2 = len(sig2.Y)
        X =sig1.X
        Y =[]
        corr = self.compute_cross_correlation(sig1.Y,sig2.Y)
        Y = self.compute_normalize_correlation(sig1.Y,sig2.Y,corr)



        self.last_signal = copy.deepcopy(Signal(X,Y))
        print(X)
        print(Y)
        self.plot_signals(X,Y,"Conv the signal")

    def convlute_signal(self):
        if len(self.stack) < 2:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig2 = copy.deepcopy(self.stack[len(self.stack)-1])
        sig1 = copy.deepcopy(self.stack[len(self.stack)-2])

        len1 = len(sig1.Y)
        len2 = len(sig2.Y)

        start_index = int(min(sig1.X) + min(sig2.X))
        end_index = int(max(sig1.X) + max(sig2.X))

        X = list(range(start_index, end_index + 1))
        Y = []
        for n in range(len1 + len2 - 1):
            sum = 0
            for m in range(min(n, len1 - 1) + 1):
                if 0 <= n - m < len2:
                    sum += sig1.Y[m] * sig2.Y[n - m]
            Y.append(sum)

        self.last_signal = copy.deepcopy(Signal(X,Y))
        print(X)
        print(Y)
        self.plot_signals(X,Y,"Conv the signal")

    def remove_DC_freq(self):
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])
        sampling_rate = float(self.FreqDFT_entry.get())
        frequencies = np.fft.fftfreq(len(sig.Y), d=1/sampling_rate)
        dc_removed_signal = self.remove_dc_component(sig.Y, frequencies)
        sig.Y = [round(x,3) for x in dc_removed_signal]

        self.last_signal = copy.deepcopy(sig)
        print(sig.Y)
        self.plot_signals(sig.X,sig.Y,"remove DC Signal freq")

    def remove_dc_component(self,signal, frequencies):
        X = self.calculate_dft(signal)

        # Identify the index corresponding to DC
        dc_index = np.argmax(frequencies >= 0)

        # Set the DC component to zero
        X[dc_index] = 0

        # Calculate inverse DFT to get the filtered signal
        dc_removed_signal = self.calculate_idft(X)
        return dc_removed_signal.real
    def move_avg_signal(self):
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])

        win = int(self.multiplication_entry.get())
        if win>len(sig.Y):
            messagebox.showinfo("invalid input","Please enter valid window size!")
            return

        sig.Y = list(itertools.accumulate(sig.Y))

        X = []
        Y = []

        for i in range(0,len(sig.Y)-win+1):
            X.append(sig.X[i])
            if i!=0:
                Y.append(round((sig.Y[i+win-1]-sig.Y[i-1])/win,3))
            else:
                Y.append(round(sig.Y[i+win-1]/win,3))
        print(X)
        print(Y)

        self.last_signal = copy.deepcopy(Signal(X,Y))

        self.plot_signals(X,Y,"Move Avg signal")

    def normalize_signal(self):
        # nwmin + (nwmax - nwmin) * ((val - oldmn) / (oldmx - oldmn))
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])
        old_min = min(sig.Y)
        old_max = max(sig.Y)
        new_min = float(self.min_value_norm_Entry.get())
        new_max = float(self.max_value_norm_Entry.get())

        sig.Y = list(map(lambda v: ((v - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min, sig.Y))
        self.last_signal = copy.deepcopy(sig)
        self.plot_signals(sig.X,sig.Y,"Normalized Signal")


    def accumulate_signal(self):
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])

        sig.Y = list(itertools.accumulate(sig.Y))

        self.last_signal = copy.deepcopy(sig)
        self.plot_signals(sig.X,sig.Y,"Accumulated Signal")


    def square_signal(self):
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])
        sig.Y = [x * x for x in sig.Y]

        self.last_signal = copy.deepcopy(sig)
        self.plot_signals(sig.X,sig.Y,"Squared signal")


    def multiply_signal_by_factor(self):
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])

        factor = float(self.multiplication_entry.get())

        sig.Y = [x * factor for x in sig.Y]
        self.last_signal = copy.deepcopy(sig)

        self.plot_signals(sig.X,sig.Y,"Multiplied signal")

        # print(len(self.last_signal.X))



    def add_two_signals(self, multiplier):
        if len(self.stack) < 2:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig2 = copy.deepcopy(self.stack[len(self.stack)-1])
        sig1 = copy.deepcopy(self.stack[len(self.stack)-2])
        i = 0
        j = 0
        tmp = Signal([],[])

        while i<len(sig1.X) and j <len(sig2.X):
            if sig1.X[i] == sig2.X[j]:
                tmp.X.append(sig1.X[i])
                tmp.Y.append(sig1.Y[i] + multiplier*sig2.Y[j])
                i = i + 1
                j = j + 1
            elif sig1.X[i] < sig2.X[j]:
                tmp.X.append(sig1.X[i])
                tmp.Y.append(sig1.Y[i])
                i = i + 1
            else:
                tmp.X.append(sig2.X[j])
                tmp.Y.append(multiplier * sig2.Y[j])
                j = j + 1

        while i<len(sig1.X):
            tmp.X.append(sig1.X[i])
            tmp.Y.append(sig1.Y[i])
            i = i + 1

        while j<len(sig2.X):
            tmp.X.append(sig2.X[j])
            tmp.Y.append(multiplier * sig2.Y[j])
            j = j + 1

        self.last_signal = copy.deepcopy(tmp)
        self.plot_signals(tmp.X,tmp.Y,"Add/Sub Signal")


    def generate_signal(self):
        if self.current_mode == 'signal':
            self.single_signal_generator()
        elif self.current_mode == 'sincos':
            self.sincos_signal_generator()
        elif self.current_mode=='process':
            self.process_signal_generator()
        elif self.current_mode=='Quantize':
            self.Quantize_signal_generator()
        elif self.current_mode =='Time Domain':
            self.Time_Domain_signal_generator()
        else:
            messagebox.showinfo("Error","chosen mode is undifiend")


    def compare_signals(self):
        compare_path = self.navigate_file()
        if self.current_mode == 'Quantize':
            if self.signal_to_quan_combobox.get() =='bit': # text 1
                self.QuantizationTest1(compare_path,self.encoded_signal,self.quantized_signal)
            else:
                self.QuantizationTest2(compare_path,self.intervals,self.encoded_signal,self.quantized_signal,self.quantization_error)
        elif self.is_derevative == 1:
            self.SignalSamplesAreEqual(compare_path,self.first_der.X,self.first_der.Y)
            compare_path = self.navigate_file()
            self.SignalSamplesAreEqual(compare_path,self.second_der.X,self.second_der.Y)
            self.is_derevative = 0
        else:
            self.SignalSamplesAreEqual(compare_path, self.last_signal.X, self.last_signal.Y)


    def plot_signals_(self,step,sig_1,sig_2):
        self.rgb_idx = (self.rgb_idx + 1 ) % 4
        plot_n = []

    def plot_signals(self, n, signal, text):
        self.rgb_idx = (self.rgb_idx + 1 ) % 4
        plot_n = []
        plot_signal = []
        t = np.linspace(0, 1, len(n), endpoint=False)  # Time for continuous plot
        plot_t = []
        mn = min(len(signal), 200)  # Limit the plot to the first 200 samples (for large signals)

        # Prepare the data to plot
        for i in range(mn):
            # print(n[i]+ " " + signal[i])
            plot_n.append(n[i])  # Discrete time (indices)
            plot_signal.append(signal[i])  # Signal values (discrete and continuous)
            plot_t.append(t[i])  # Continuous time

        # Plot continuous signal
        plt.subplot(2, 1, 1)
        plt.plot(plot_t, plot_signal, label=text + '(continues)', color=self.RGB[self.rgb_idx])
        plt.title("Continuous Signal")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.legend()

        # Plot discrete signal
        plt.subplot(2, 1, 2)
        plt.stem(plot_n, plot_signal, label=text + '(discrete)', linefmt='black', markerfmt=self.RGB[self.rgb_idx][0]+'o', basefmt='r-')
        plt.title("Discrete Signal")
        plt.xlabel('Indices')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.legend()

        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()

    def Quantize_signal_generator(self):
        sig =  copy.deepcopy(self.stack[len(self.stack)-1])
        # for y in sig.Y:
        #     print (y)
        max_value = float(max(sig.Y))
        min_value = float(min(sig.Y))
        delta = max_value - min_value

        levels = int(self.quantize_entry.get())
        bits = levels
        if(self.signal_to_quan_combobox.get() == 'bit'):
            levels = 2 ** levels
        else:
            bits = int(math.log2(levels))

        print("levels:", levels, "min: ", min_value,"max: ",max_value)
        delta = round(float(delta / (levels)),3)
        print("delta: ",delta)
        intervals_ = []
        cur_val = round(min_value,3)
        while cur_val <max_value:
            intervals_.append([round(cur_val,3),round(cur_val+delta,3)])
            cur_val+=round(delta,3)
            cur_val= round(cur_val,3)
        #
        # for [l,r] in intervals_:
        #     print(l,r)
        #(compare_path,self.intervals,self.encoded_signal,self.quantized_signal,self.quantization_error)
        self.intervals =[]
        self.quantization_error = []
        self.quantized_signal =[]
        self.encoded_signal =[]
        for y in sig.Y:
            near = float(2**20)
            self.intervals.append(0)
            self.quantization_error.append(0)
            self.quantized_signal.append(0)
            self.encoded_signal.append('0')
            for i in range(0,len(intervals_)):
                # print(intervals_[i][0],"->",y," <- ",intervals_[i][1])
                mid_point = round((intervals_[i][1] + intervals_[i][0])/2,3)

                if round(abs(mid_point - y),3)<near:

                    near = round(abs(mid_point - y),3)
                    self.intervals[len(self.intervals)-1] = i+1
                    self.encoded_signal[len(self.encoded_signal)-1] = bin(i)[2:].zfill(bits)
                    self.quantization_error [len(self.quantization_error)-1] = round(mid_point - y,3)
                    self.quantized_signal [len(self.quantized_signal)-1] = mid_point

        for i in range(len(sig.Y)):
            print(self.encoded_signal[i], self.quantized_signal[i],sig.Y[i])



        print("signal generated succ!")


    def convert_from_polar(self,sig):
        tmpX = []
        tmpY = []

        for i in range(len(sig.X)):
            tmpY.append(round(sig.X[i]*math.sin(sig.Y[i]),13))
            tmpX.append(round(sig.X[i]*math.cos(sig.Y[i]),13))

        sig = Signal(tmpX,tmpY)
        return sig
    #[0j, (1.1102230246251565e-15+6.000103061172299j),


    def DFT_signal(self, div=1, mul=-1):
        sig = copy.deepcopy(self.stack[len(self.stack)-1])
        N = len(sig.Y)
        if div !=1:
            div = N
            sig = self.convert_from_polar(sig)
            #print(sig.X)
            #print(sig.Y)


        result = []

        if mul == -1:
            for k in range(0,N):
                result.append(0)
                for n in range(0,N):
                    tmp =  sig.Y[n] #x(n)
                    angle = (2*180*k*n)/N
                    angle = math.radians(angle)
                    tmp = tmp*(math.cos(angle)+mul*(math.sin(angle)*1j))
                    result[len(result)-1] += tmp

                result[k] = result[k]/div
        else:
            for n in range(0,N):
                result.append(0)
                for k in range(0,N):
                    tmp = sig.X[k] + sig.Y[k]*1j #x(n)
                    #print(sig.X[k],sig.Y[k],tmp)
                    #print(tmp)
                    angle = (2*180*k*n)/N
                    angle = math.radians(angle)
                    tmp = tmp*(math.cos(angle)+mul*(math.sin(angle)*1j))
                    result[len(result)-1] += tmp

                result[n] = result[n]/div

        if div == 1:
            ampl = []
            phase =[]


            for comp in result:
                ampl.append(abs(comp))
                #print(comp.real,comp.imag)
                phase.append(math.atan2(comp.imag,comp.real))

            for i in range(0,N):
                print(ampl[i],phase[i])

            Fs = float(self.FreqDFT_entry.get())
            omega = (2*math.pi*Fs) / (N)

            X = []
            for i in range(0,len(ampl)):
                X.append((i+1)*omega)

            self.plot_signals(X,ampl,"Time vs ampl")
            self.plot_signals(X,phase,"Time vs phase")


            # self.debug(ampl)
            # self.debug(phase)
        else:
            idx = 0
            for n in result:
                print(idx , round(n.real))
                idx = idx+ 1

    def calculate_dft(self,signal):
        N = len(signal)
        X = np.zeros(N, dtype=complex)
        for k in range(N):
            X[k] = 0
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        return X

    def calculate_idft(self,X):
        N = len(X)
        signal = np.zeros(N, dtype=complex)

        for n in range(N):
            signal[n] = 0
            for k in range(N):
                angle = 2 * np.pi * n * k / N
                real_part = np.cos(angle)
                imaginary_part = np.sin(angle)
                signal[n] += (X[k].real * real_part) - (X[k].imag * imaginary_part)

            signal[n] /= N

        return np.asarray(signal, float)

    def debug(self, lst):
        for i in lst:
            print (i)

    def DCT_signal(self):
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])

        N = len(sig.Y)
        Y = []
        for k in range(N ):  # k ranges from 1 to N
            coeff = 0
            for n in range(N):  # n ranges from 1 to N
                coeff += sig.Y[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
            coeff *= math.sqrt(2 / N)  # Multiply by sqrt(2/N)
            Y.append(coeff)
            sig.X[k] = 0

        sig.Y = copy.deepcopy(Y)
        self.last_signal = copy.deepcopy(sig)
        #print(Y)
        self.plot_signals(sig.X,sig.Y,"DCT Signal")

    def Time_Domain_signal_generator(self):
        option = self.time_domain_options_combobox.get()
        if option == 'sharp':
            self.sharp_signal_generator()
        elif option =='fold':
            self.fold_signal_generator()
        elif option == 'shift':
            self.shift_signal_generator()
        elif option == 'fold and shift':
            self.fold_and_shift_generator()
        elif option =='remove DC':
            self.remove_DC_time()
        elif option =='filter':
            self.fir_filter_generator()
        elif option=='resample':
            self.resample_generator()
        else:
            print("error in Time_Domain_signal_generator method!!")

    def upsample(self,signal, factor):
        result = []
        for element in signal:
            result.extend([element] + [0] * (factor-1))
        for i in range(factor-1):
            result.pop()
        # print (result)
        return result
    def resample_signal(self,input_x, input_y, M, L, filter_type, fs, stop_band_attenuation, transition_band,f1,f2 = None):
        print(M,L)
        if M == 0 and L != 0:
            # Upsample by inserting L-1 zeros between each sample
            upsampled_signal = self.upsample(input_y, L)
            upsampled_x = self.upsample(input_x, L)
            upsampled_x = list(range(min(upsampled_x), min(upsampled_x) + len(upsampled_x)))
            print(upsampled_signal)
            filtered_x = []
            filtered_y = []
            if f2 == None:
                filtered_x, filtered_y = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band,f1)
            else:
                filtered_x, filtered_y = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band,f1,f2)

            return self.calc_convolution(upsampled_x, upsampled_signal, filtered_x, filtered_y)

        elif M != 0 and L == 0:
            # Downsample by taking every Mth sample
            filtered_x = []
            filtered_y = []
            if f2 == None:
                filtered_x, filtered_y = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band,f1)
            else:
                filtered_x, filtered_y = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band,f1,f2)
            output_x, output_y = self.calc_convolution(input_x, input_y, filtered_x, filtered_y)
            output_x, output_y = output_x[::M], output_y[::M]

            continuous_indices = list(range(min(output_x), min(output_x) + len(output_x)))

            return continuous_indices, output_y


        elif M != 0 and L != 0:
            # Upsample, filter, and then downsample
            upsampled_signal = self.upsample(input_y, L)
            upsampled_x = self.upsample(input_x, L)
            upsampled_x = list(range(min(upsampled_x), min(upsampled_x) + len(upsampled_x)))
            filtered_x = []
            filtered_y = []
            if f2 == None:
                filtered_x, filtered_y = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band,f1)
            else:
                filtered_x, filtered_y = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band,f1,f2)
            filtered_signal_x, filtered_signal_y = self.calc_convolution(upsampled_x, upsampled_signal, filtered_x, filtered_y)
            filtered_signal_x, filtered_signal_y = filtered_signal_x[::M], filtered_signal_y[::M]
            print(filtered_signal_y)

            continuous_indices = list(range(min(filtered_signal_x), min(filtered_signal_x) + len(filtered_signal_x)))

            return continuous_indices, filtered_signal_y

        else:
            return messagebox.showerror("Invalid values for M and L")
    def resample_generator(self):
        filter_type = (self.time_domain_filter_combobox.get())
        fs = float(self.sampling_freq_entry.get())
        stop_band_attenuation = float(self.stop_band_att_entry.get())
        f1 = float(self.FC1_entry.get())
        if filter_type == "Band pass" or filter_type == "Band stop":
            f2 = float(self.FC2_entry.get())
        transition_band = float(self.trans_band_entry.get())
        M = int(self.decimation_M_entry.get())
        L = int(self.interpolate_L_entry.get())

        if len(self.stack) ==0:
            return messagebox.showerror("NO enough resources!")

        sig = copy.deepcopy(self.stack[-1])
        sig.X  = [int(x) for x in sig.X]
        sig.Y  = [int(x) for x in sig.Y]

        if filter_type == "Low pass" or filter_type == "High pass":
            resample_res_x, resample_res_y  = self.resample_signal(sig.X, sig.Y, M, L, filter_type, fs, stop_band_attenuation, transition_band,f1)
        else:
            resample_res_x, resample_res_y  = self.resample_signal(sig.X, sig.Y, M, L, filter_type, fs, stop_band_attenuation, transition_band,f1,f2)

        self.last_signal.X = copy.deepcopy(resample_res_x)
        self.last_signal.Y = copy.deepcopy(resample_res_y)

        self.plot_signals(resample_res_x,resample_res_y,"resampled signal")

    def round_up_to_odd(self,number):
        rounded_number = math.ceil(number)

        if rounded_number % 2 == 0:
            rounded_number += 1

        return rounded_number


    def window_function(self,stop_band_attenuation, n, N):
        if stop_band_attenuation <= 21:     # Rectangular
            return 1
        elif stop_band_attenuation <= 44:   # Hanning
            return 0.5 + (0.5 * np.cos((2 * np.pi * n) / N))
        elif stop_band_attenuation <= 53:   # Hamming
            return 0.54 + (0.46 * np.cos((2 * np.pi * n) / N))
        elif stop_band_attenuation <= 74:   # Blackman
            return 0.42 + (0.5 * np.cos(2 * np.pi * n / (N - 1))) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    def init_fir_filter(self,filter_type, fs, stop_band_attenuation, transition_band, f1, f2=None):
        delta_f = transition_band / fs
        print(filter_type,fs,stop_band_attenuation,transition_band,f1)
        N = 0
        if stop_band_attenuation <= 21:     # Rectangular
            N = self.round_up_to_odd(0.9/delta_f)
        elif stop_band_attenuation <= 44:   # Hanning
            N = self.round_up_to_odd(3.1/delta_f)
        elif stop_band_attenuation <= 53:   # Hamming
            N = self.round_up_to_odd(3.3/delta_f)
        elif stop_band_attenuation <= 74:   # Blackman
            N = self.round_up_to_odd(5.5/delta_f)

        print(N)
        # the list to hold the filter
        h = []
        # the x values
        indices = range(-math.floor(N/2), math.floor(N/2) + 1)

        # get filter
        #   low-pass
        if filter_type == 'Low pass':
            new_fc = f1 + 0.5 * transition_band
            new_fc = new_fc / fs

            for n in indices:
                w_n = self.window_function(stop_band_attenuation, n, N)
                # print("win", w_n)
                if n == 0:
                    h_d = 2*new_fc
                else:
                    h_d = 2*new_fc * (np.sin(n*2*np.pi*new_fc)/(n*2*np.pi*new_fc))
                    # print("hd",h_d)
                    # print(new_fc)
                h.append(h_d*w_n)
            # [print(row) for row in list(zip(indices, h))]

        #   high-pass
        elif filter_type == 'High pass':
            new_fc = f1 - 0.5 * transition_band
            new_fc /= fs

            for n in indices:
                w_n = self.window_function(stop_band_attenuation, n, N)
                if n == 0:
                    h_d = 1 - 2*new_fc
                else:
                    h_d = -2*new_fc * (np.sin(n*2*np.pi*new_fc)/(n*2*np.pi*new_fc))
                h.append(h_d * w_n)
            # [print(row) for row in list(zip(indices, h))]

        #   band-pass
        elif filter_type == 'Band pass':
            new_fc = f1 - 0.5 * transition_band
            new_fc /= fs
            new_fc2 = f2 + 0.5 * transition_band
            new_fc2 /= fs

            for n in indices:
                w_n = self.window_function(stop_band_attenuation, n, N)
                if n == 0:
                    h_d = 2*(new_fc2 - new_fc)
                else:
                    h_d = 2*new_fc2*(np.sin(n*2*np.pi*new_fc2)/(n*2*np.pi*new_fc2)) - 2*new_fc*(np.sin(n*2*np.pi*new_fc)/(n*2*np.pi*new_fc))
                h.append(h_d * w_n)
            # [print(row) for row in list(zip(indices, h))]

        #   band-stop
        elif filter_type == 'Band stop':
            new_fc = f1 + 0.5 * transition_band
            new_fc /= fs
            new_fc2 = f2 - 0.5 * transition_band
            new_fc2 /= fs

            for n in indices:
                w_n = self.window_function(stop_band_attenuation, n, N)
                if n == 0:
                    h_d = 1-2*(new_fc2 - new_fc)
                else:
                    h_d = 2*new_fc*(np.sin(n*2*np.pi*new_fc)/(n*2*np.pi*new_fc)) - 2*new_fc2*(np.sin(n*2*np.pi*new_fc2)/(n*2*np.pi*new_fc2))
                h.append(h_d * w_n)
            # [print(row) for row in list(zip(indices, h))]

        return indices, h


    def calc_convolution(self,x_values1, y_values1, x_values2, y_values2):
        len1 = len(y_values1)
        len2 = len(y_values2)
        result = []

        start_index = int(min(x_values1) + min(x_values2))
        end_index = int(max(x_values1) + max(x_values2))

        x_values = list(range(start_index, end_index + 1))

        for n in range(len1 + len2 - 1):
            sum = 0
            for m in range(min(n, len1 - 1) + 1):
                if 0 <= n - m < len2:
                    sum += y_values1[m] * y_values2[n - m]
            result.append(sum)
        return x_values, result

    def fir_filter_generator(self):
        filter_type = (self.time_domain_filter_combobox.get())
        fs = float(self.sampling_freq_entry.get())
        stop_band_attenuation = float(self.stop_band_att_entry.get())
        f1 = float(self.FC1_entry.get())
        if filter_type == "Band pass" or filter_type == "Band stop":
            f2 = float(self.FC2_entry.get())
        transition_band = float(self.trans_band_entry.get())

        if filter_type == "Low pass" or filter_type == "High pass":
            indices, filter_ = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band,f1)
        else:
            indices, filter_ = self.init_fir_filter(filter_type, fs, stop_band_attenuation, transition_band, f1, f2)

        self.plot_signals(indices,filter_,"filter coff")

        if len(self.stack)!=0:
            sig = copy.deepcopy(self.stack[-1])
            X,Y = self.calc_convolution(sig.X,sig.Y,indices,filter_)
            self.plot_signals(X,Y,"filtered signal")

            self.last_signal.X = copy.deepcopy(X)
            self.last_signal.Y = copy.deepcopy(Y)
        else:
            self.last_signal.X = copy.deepcopy(indices)
            self.last_signal.Y = copy.deepcopy(filter_)



    def remove_DC_time(self):
        if len(self.stack)==0:
            messagebox.showinfo("Low resources","Please load enough signals!")
            return

        sig = copy.deepcopy(self.stack[len(self.stack)-1])
        mean_value = round(sum(sig.Y)/len(sig.Y),3)
        sig.Y = [round(x - mean_value,3) for x in sig.Y]
        self.last_signal = copy.deepcopy(sig)
        print(sig.Y)
        self.plot_signals(sig.X,sig.Y,"Remove DC Time Signal")




    def sharp_signal_generator(self):
     self.is_derevative = 1
     if len(self.stack) == 0:
        messagebox.showinfo("Low resources", "Please load enough signals!")
        return

     sig = copy.deepcopy(self.stack[-1])

     self.first_der.clear()
     self.second_der.clear()

     for i in range(1,len(sig.Y)):
         self.first_der.Y.append(sig.Y[i]- sig.Y[i-1])
         self.first_der.X.append(sig.X[i])

     for i in range(1,len(sig.Y)-1):
         self.second_der.Y.append(sig.Y[i+1]- 2*sig.Y[i]+sig.Y[i-1])
         self.second_der.X.append(sig.X[i])


     self.last_signal = copy.deepcopy(sig)

     self.plot_signals(self.first_der.X, self.first_der.Y, "First Derivative Signal")
     self.plot_signals(self.second_der.X, self.second_der.Y, "Second Derivative Signal")



    def fold_signal_generator(self):
     if len(self.stack) == 0:
        messagebox.showinfo("Low resources", "Please load enough signals!")
        return

     sig = copy.deepcopy(self.stack[-1])

     n = len(sig.Y)
     folded_Y = [0] * n
     for i in range(n):
        folded_Y[i] = sig.Y[n - i - 1]


     sig.Y = copy.deepcopy(folded_Y)
     ##sig.X = folded_X
     self.last_signal = copy.deepcopy(sig)
     #print(folded_Y)
     self.plot_signals(sig.X, sig.Y, "Folded Signal")






    def shift_signal_generator(self):

     if len(self.stack) == 0:
        messagebox.showinfo("Low resources", "Please load enough signals!")
        return


     shftamt = int(self.shift_entry.get())


     sig = copy.deepcopy(self.stack[-1])


     shifted_X = [0] * len(sig.X)
     for i in range(len(sig.X)):
        shifted_X[i] = sig.X[i] + shftamt


     sig.X = copy.deepcopy(shifted_X)
     self.last_signal = copy.deepcopy(sig)

     self.plot_signals(sig.X, sig.Y, "Shifted Signal")





    def fold_and_shift_generator(self):
     shftamt = int(self.shift_entry.get())
     if len(self.stack) == 0:
        messagebox.showinfo("Low resources", "Please load enough signals!")
        return

     sig = copy.deepcopy(self.stack[-1])

     n = len(sig.Y)
     folded_Y = [0] * n
     for i in range(n):
        folded_Y[i] = sig.Y[n - i - 1]

     sig.Y = copy.deepcopy(folded_Y)

     shifted_X = [0] * len(sig.X)
     for i in range(len(sig.X)):
        shifted_X[i] = sig.X[i] + shftamt


     sig.X = copy.deepcopy(shifted_X)

     self.last_signal = copy.deepcopy(sig)

     self.plot_signals(sig.X, sig.Y, "Folded and Shifted Signal")




if __name__ == "__main__":
    root = tk.Tk()
    root.title("Signal Generator")
    root.geometry("400x800")
    app = Task_1(root)
    root.mainloop()
