import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses

import pandas as pd
import numpy as np
from sys import stderr
from pprint import pprint

UPPER_BOUND = 0.7
LOWER_BOUND = 0.3
def map_to_discreet(arr: list[float]) -> list[int]:
    res = list()
    for num in arr:
        if num >= UPPER_BOUND:
            res.append(1)
        elif num <= LOWER_BOUND:
            res.append(0)
        else:
            res.append(None)
    return res

def train_with_steps(model: Sequential, x: pd.DataFrame, y: pd.DataFrame, episodes: int, step: int, **kwargs) -> None:
    if episodes % step != 0:
        raise Exception("Episodes must be divisible by step")

    mse = losses.MeanSquaredError()

    steps_count = episodes // step
    for i_step in range(steps_count):
        model.fit(x, y, epochs=step,  **kwargs) #initial_epoch=i_step * step,
        pred = model.predict(x, verbose=0)
        avg_mse = mse(y, pred)
        #rel = map(lambda row: , x.iterrows())
        print(f'Epoch {i_step * step}/{episodes}: MSE -> {avg_mse * 100: .3f}')
        #print(f'Epoch {i_step * step}/{episodes}: accuracy -> {avg_mse * 100: .3f}%', file=stderr)
        # TODO: continue here (evaluation -> reliability on same data)


def main() -> None:
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # DONE: create DataFrame
    data = [
        [0,0,0,0], #
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1], #
        [1,1,0,0], #
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0], #
        [1,0,1,1],
        [0,1,0,0],
        [0,1,0,1], #
        [0,1,1,0],
        [0,1,1,1]
    ]
    df = pd.DataFrame(data)
    print('Created input data')

    # DONE: create model
    model = Sequential()
    model.add(
        layers.Dense(2, input_shape=(4,), activation='relu')
    )
    model.add(
        layers.Dense(4, activation='sigmoid')
    )
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    print('Model has been created')

    # DONE: choose order of training patterns
    # sequential

    # DONE: train model
    train_with_steps(model, df, df, 10_000, 500, shuffle=True, verbose=0, workers=4)
    #model.fit(x=df, y=df, epochs=8_000, shuffle=False, verbose=1, workers=4)

    # DONE: discuss results
    predictions = model.predict(df, verbose=0)
    print()
    pairs = [ (d[1].tolist(), p) for d, p in zip(df.iterrows(), predictions) ]
    print('Printing results of prediction using format: [input] -> [prediction]')
    successful_pred = 0
    avg_acc, avg_rel = list(), list()
    full_acc_counter = 0
    for dat, pred in pairs:
        pred_discrete = map_to_discreet(pred)
        pred_closest = list(map(round, pred))
        accuracy, reliability = 0, 0
        for val1, val2, val3 in zip(dat, pred, pred_discrete):
            if val3 is not None:
                reliability += 1
                accuracy += int(abs(val1 - val2) < 0.5)
        successful_pred += int(dat == pred_closest)
        print(f'{dat} --> {pred}')
        print(f'>>> (mapped with bounds) {pred_discrete}')
        print(f'>>> (mapped to closest) { pred_closest }')

        if reliability == 0:
            print(f'>>> reliability: 0%')
            avg_acc.append(0)
            avg_rel.append(0)
            continue

        print(f'>>> acc: {accuracy * 100 / len(dat): .2f}%\t reliability: { reliability * 100 / len(dat) : .2f}%')
        #print(f'{dat} --> {pred} (mapped) {pred_discrete} -> Success? {dat == map_to_discreet(pred)}', file=stderr)

        if accuracy == len(dat):
            full_acc_counter += 1

        avg_acc.append(accuracy / len(dat))
        avg_rel.append( reliability / len(dat) )

    print(f'Successful predictions rate: {successful_pred}/{len(predictions)}')
    print(f'Average accuracy: {np.mean(avg_acc) * 100: .2f}%')
    print(f'Average reliability: {np.mean(avg_rel) * 100: .2f}%')
    print(f'Fully accurate counter: {full_acc_counter}')
    #print(f'Successful predictions rate: {successful_pred}/{len(predictions)}', file=stderr)

if __name__ == '__main__':
    main()