import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from perceptron import Perceptron


def main():
    #  getting data from kaggle's csv dataset
    data = pd.read_csv('mushrooms.csv')

    #  changing classes p and e to 1 and 0
    data['class'] = [1 if i == "p" else 0 for i in data["class"]]

    #  get rid of "veil-type" column
    data.drop('veil-type', axis=1, inplace=True)

    # replace letters in each column to the numbers
    for column in data.drop(["class"], axis=1).columns:
        value = 0
        step = 1 / (len(data[column].unique()) - 1)
        for i in data[column].unique():
            data[column] = [value if letter == i else letter for letter in data[column]]
            value += step

    x_data = data.drop('class', axis=1).values
    y_data = data['class'].values

    perceptron = Perceptron(21)

    # train perceptron
    perceptron.train(x_data, y_data)

    # test perceptron for predictions in random 10 mushrooms
    random_mushrooms = data.sample(10)
    for index, mushroom in random_mushrooms.iterrows():
        expected_value = ['Poisonous' if mushroom['class'] == 0 else 'Edible']
        print(index+1, 'Expected value: ', expected_value[0])
        prediction_value = ['Poisonous' if perceptron.predict(mushroom.drop(['class'])) == 0 else 'Edible']
        print(index+1, 'Prediction: ', prediction_value[0], '\n')


if __name__ == '__main__':
    main()
