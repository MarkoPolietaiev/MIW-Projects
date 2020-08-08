import sys
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from sygnals import Signal, generate_sampling


def display_time_series(x, y):
    plt.plot(x, y)
    plt.show()


def get_windowed_data(window_size, step, data):
    start = 0
    end = start + window_size
    x = []
    y = []
    while end < data.shape[0]:
        x.append(data[start:end])
        y.append(data[end])
        start += step
        end = start + window_size
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def reshape_input_for_lstm(x):
    # ILOŚĆ ELEMENTÓW, ROZMIAR_OKNA, WYMIAR_WEJŚCIA
    return np.reshape(x, (x.shape[0], x.shape[1], 1))


# model.add(tf.keras.layers.LSTM(input_size,
#                               dropout=0.2,
#                               recurrent_dropout=0.2,
#                               return_sequences=True,
#                               activation='relu',
#                               kernel_regularizer=regularizers.l2(0.001),
#                               stateful=False,
#                               input_shape=input_shape
# ))


def create_neural_network(x, window_size):
    window_size = x.shape[1]
    input_layer = tf.keras.layers.LSTM(window_size, activation='relu', input_shape=(window_size, 1),
                                       return_sequences=True)
    hidden_layer = tf.keras.layers.LSTM(window_size, activation='relu', return_sequences=True)
    output_layer = tf.keras.layers.LSTM(1, activation='linear')
    network = tf.keras.Sequential([input_layer,
                                   hidden_layer,
                                   output_layer])
    network.compile(loss='mse', optimizer='rmsprop')
    return network


def show_prediction_and_reference(pred, ref):
    plt.plot(ref, color='blue', label='Real signal values')
    plt.plot(pred, color='red', label='Predicted Signal values')
    plt.title('Signal prediction with LSTM')
    plt.legend()
    plt.show()


def predict(model, windowed_timeseries, prediction_depth):
    predictions = model.predict(windowed_timeseries)
    processed = prediction_depth // len(windowed_timeseries)
    depth = prediction_depth
    while processed >= 0:
        if processed > 0:
            depth -= len(windowed_timeseries)
            future_signals = model.predict(windowed_timeseries[:depth])
            predictions = np.vstack((predictions, future_signals))
        else:
            future_signals = model.predict(windowed_timeseries[:depth])
            predictions = np.vstack((predictions, future_signals))
        processed -= 1
    return predictions


def main(signal_complexity, window_size, window_step, noise_amplitude, prediction_depth, display_only,
         epoch_count):
    signal = Signal.generate_random_signal((1, 20), (1, 20), (0, 2 * np.pi), (0, 0), signal_complexity, noise_amplitude)

    sampling_tr = generate_sampling(0, 10, 2000)
    sampling_val = generate_sampling(10, 20, 2000)

    y_tr = signal.generate_timeseries(sampling_tr)
    display_time_series(sampling_tr, y_tr)
    if display_only:
        sys.exit()
    x_tr, y_tr = get_windowed_data(window_size, window_step, y_tr)
    x_tr = reshape_input_for_lstm(x_tr)

    y_val = signal.generate_timeseries(sampling_val)
    x_val, y_val = get_windowed_data(window_size, window_step, y_val)
    x_val = reshape_input_for_lstm(x_val)

    network = create_neural_network(x_tr, window_size)
    history = network.fit(x_tr, y_tr,
                          epochs=epoch_count, shuffle=False, batch_size=32,
                          validation_data=(x_val, y_val))

    sampling_prediction_input = generate_sampling(0, 30, 2000)
    y_pred = signal.generate_timeseries(sampling_prediction_input)
    x_pred, _ = get_windowed_data(window_size, window_step, y_pred)
    x_pred = reshape_input_for_lstm(x_pred)

    sampling_prediction_results = generate_sampling(30, 60, 2000)
    sampling_prediction_results = sampling_prediction_results[:prediction_depth]
    prediction_reference = signal.generate_timeseries(sampling_prediction_results)

    predictions = predict(network, x_pred, prediction_depth)
    show_prediction_and_reference(predictions, prediction_reference)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--signal_complexity', type=int, default=0)
    parser.add_argument('-e', '--epoch_count', type=int, default=1)
    parser.add_argument('-n', '--noise_amplitude', type=float, default=0.0)
    parser.add_argument('-w', '--window_size', type=int, default=5)
    parser.add_argument('-s', '--window_step', type=int, default=1)
    parser.add_argument('-d', '--prediction_depth', type=int, required=True)
    parser.add_argument('--display_only', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.signal_complexity, args.window_size, args.window_step, args.noise_amplitude, args.prediction_depth,
         args.display_only,
         args.epoch_count)
    #  -c 1 -n 10  -d 4000 -e 2
    #  -c 0 -d 2000 -e 1
    #  -c 3 -n 20  -d 4000 -e 10
