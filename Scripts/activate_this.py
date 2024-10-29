import math
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
                x, y = re.split(r",\s*",lines[i].strip())
                X.append(float(x.rstrip("f")))
                Y.append(float(y.rstrip("f")))

            signal_tmp = Signal(X, Y)
            return signal_tmp


class Signal:
    def __init__(self, x, y):
        self.X = x
        self.Y = y


class Task_1:
    def __init__(self, root):
        self.root = root
        self.create_widgets()
        self.stack: List[Signal([],[])] = []  # stack of signals
        self.last_signal = Signal([], [])
        self.current_mode = "signal"

        self.RGB = ['red','green','blue','yellow']
        self.rgb_idx = 0

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
                                                                      'normalize','DFT','IDFT'])
        self.signal_processing_options_combobox.bind("<<ComboboxSelected>>", self.on_processing_options_change)

        self.signal_processing_options_combobox.set('square')

        self.multiplication_entry = tk.Entry(self.root)
        self.multiplication_entry.insert(0, "0.0")

        self.FreqDFT_entry = tk.Entry(self.root)
        self.FreqDFT_entry.insert(0, "0.0")

        self.signal_to_draw_label = tk.Label(self.root, text="Signal to draw:", bg='lightblue', fg='darkblue')
        self.signal_to_draw_combobox = ttk.Combobox(self.root, values=['signal', 'sincos signal', 'processed signal','Quantize'])
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


    def on_processing_options_change(self,event):
        # Get the selected value
        selected_value = self.signal_processing_options_combobox.get()
        # ['add', 'sub', 'multiply', 'square', 'accumulate', 'normalize']
        if selected_value == 'multiply':
            self.multiplication_entry.pack(pady=2)
        else:
            self.multiplication_entry.pack_forget()
            if selected_value == 'normalize':
                self.min_value_norm_abel.pack(pady=5)
                self.min_value_norm_Entry.pack(pady=2)
                self.max_value_norm_abel.pack(pady=5)
                self.max_value_norm_Entry.pack(pady=2)
            else:
                self.min_value_norm_abel.pack_forget()
                self.min_value_norm_Entry.pack_forget()
                self.max_value_norm_abel.pack_forget()
                self.max_value_norm_Entry.pack_forget()
                if selected_value =='DFT':
                    self.FreqDFT_entry.pack(pady=5)
                else:
                    self.FreqDFT_entry.pack_forget()

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
        self.signal_to_draw_label.pack(pady=5)
        self.signal_to_draw_combobox.pack(pady=5)
        self.signal_processing_options_label.pack(pady=5)
        self.signal_processing_options_combobox.pack(pady=5)

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
            self.DFT_signal(0,-1)
        else:
            messagebox.showinfo("Option Error", "The seleceted option is undefined!")

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
        else:
            messagebox.showinfo("Error","chosen mode is undifiend")


    def compare_signals(self):
        compare_path = self.navigate_file()
        if self.current_mode == 'Quantize':
            if self.signal_to_quan_combobox.get() =='bit': # text 1
                self.QuantizationTest1(compare_path,self.encoded_signal,self.quantized_signal)
            else:
                self.QuantizationTest2(compare_path,self.intervals,self.encoded_signal,self.quantized_signal,self.quantization_error)
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
        mn = min(len(n), 200)  # Limit the plot to the first 200 samples (for large signals)

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

    def DFT_signal(self, div=1, mul=1):
        sig = copy.deepcopy(self.stack[len(self.stack)-1])
        N = len(sig.Y)
        if div !=1:
            div = N

        result = []
        for k in range(0,N):
            result.append(0)
            for n in range(0,N):
                tmp = sig.Y[n] #x(n)
                angle = (2*180*k*n)/N
                angle = math.radians(angle)
                tmp = tmp*(math.cos(angle)+mul*(math.sin(angle)*1j))
                result[len(result)-1] += tmp

            result[k] = result[k]/div

        if div == 1:
            ampl = []
            phase =[]

            for comp in result:
                ampl.append(abs(comp))
                phase.append(math.atan2(comp.imag,comp.real))

            Fs = float(self.FreqDFT_entry.get())
            omega = (2*math.pi) / (N*Fs)
            self.plot_signals()
            # self.plot_signals()

            # self.debug(ampl)
            # self.debug(phase)
        else:
            self.debug(result)
    def debug(self, lst):
        for i in lst:
            print (i)




if __name__ == "__main__":
    root = tk.Tk()
    root.title("Signal Generator")
    root.geometry("400x600")
    app = Task_1(root)
    root.mainloop()
