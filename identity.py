import tensorflow as tf
from tensorflow.keras import Sequential, layers

import pandas as pd
import numpy as np
from pprint import pprint

def main() -> None:
    # DONE: create DataFrame
    data = [
        [1,1,0,0],
        [0,0,1,1],
        [1,0,1,0],
        [0,1,0,1],
        [0,0,0,0]
    ]
    df = pd.DataFrame(data)
    print('Created input data')

    # DONE: create model
    model = Sequential()
    model.add(
        layers.Dense(2, input_shape=(4,), activation='sigmoid')
    )
    model.add(
        layers.Dense(4, activation='sigmoid')
    )
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print('Model has been created')

    # DONE: choose order of training patterns
    # sequential

    # DONE: train model
    model.fit(x=df, y=df, epochs=5_000, shuffle=False, verbose=1)

    # DONE: discuss results
    predictions = model.predict(df, verbose=0)
    print()
    pairs = [ (d, p) for d, p in zip(df.iterrows(), predictions) ]
    print('Printing results of prediction using format: [input] -> [prediction]')
    for dat, pred in pairs:
        dat_list = dat[1].tolist()
        print(f'{dat_list} --> {pred}')

if __name__ == '__main__':
    main()