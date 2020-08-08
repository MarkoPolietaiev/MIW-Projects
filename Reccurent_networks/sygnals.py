import json
import numpy as np
#import matplotlib.pyplot as plt


def generate_sampling(start_timestamp, end_timestamp, frequency):
    sample_count = (end_timestamp - start_timestamp) * frequency
    sampling = np.linspace(start_timestamp, end_timestamp, sample_count)
    return sampling


class Sinusoid:

    def __init__(self, amplitude, frequency, phase_shift, amplitude_shift):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.amplitude_shift = amplitude_shift


    def generate_timeseries(self, sampling):
        time_series = self.amplitude * np.sin(2*np.pi*self.frequency*sampling + self.phase_shift) + self.amplitude_shift
        return time_series

    @staticmethod
    def generate_random_sinusoid(amplitude, frequency, phase_shift, amplitude_shift):
        a = np.random.uniform(amplitude[0], amplitude[1])
        f = np.random.uniform(frequency[0], frequency[1])
        shift = np.random.uniform(amplitude_shift[0], amplitude_shift[1])
        phase = np.random.uniform(phase_shift[0], phase_shift[1])
        return Sinusoid(a, f, phase, shift)

    def to_json(self):
        return json.dumps(vars(self))


class Noise:

    def __init__(self, amplitude):
        self.amplitude = amplitude

    def generate_timeseries(self, sampling):
        series = np.random.uniform(0.0-self.amplitude,self.amplitude,sampling.shape[0])
        for x in range(1,series.shape[0]):
            series[x] = self.__interpolation(series[x-1], series[x], 0.03)
        return series

    @staticmethod
    def __interpolation(a, b, weight):
        return (1.0 - weight) * a + weight * b


class Signal:

    def __init__(self, carrier, signals, noise):
        self.carrier = carrier
        self.signals = signals
        self.noise = noise

    def generate_timeseries(self, sampling):
        time_series = self.carrier.generate_timeseries(sampling)
        for signal in self.signals:
            time_series += signal.generate_timeseries(sampling)
        if self.noise:
            time_series += self.noise.generate_timeseries(sampling)
        return time_series

    @staticmethod
    def generate_random_signal(amplitude, frequency, phase_shift, amplitude_shift, signal_count, noise_amplitude=0.0):
        carrier = Sinusoid(amplitude[1], frequency[0], 0, 0)
        signals = []
        noise = None
        for _ in range(signal_count):
            signals.append(Sinusoid.generate_random_sinusoid(amplitude, frequency, phase_shift, amplitude_shift))
        if noise_amplitude != 0:
            noise = Noise(noise_amplitude)
        return Signal(carrier, signals, noise)

    def to_json(self):
        contents = {'carrier':vars(self.carrier)}
        serialized_signals = []
        for signal in self.signals:
            serialized_signals.append(vars(signal))
        contents['signals'] = serialized_signals
        return json.dumps(contents)



#sinusoida = Sinusoid(1,1,0,0)
#print(sinusoida.to_json())
#sygnal = Signal.generate_random_signal((1,20), (1,3000), (0,3), (0,3), 7)
#print(sygnal.to_json())
