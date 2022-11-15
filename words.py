import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses

import pandas as pd
import numpy as np
from pprint import pprint
from itertools import chain

def transform(word: str, word_len: int) -> list[int]:
    if word == ' ':
        return [0] * (5 * word_len)

    result = list()
    for c in word:
        index = ord(c) - ord('a') + 1
        index_bin = bin(index)[2:]
        index_bin_extended = '0' * word_len + index_bin
        index_bin_final = list(map(int, index_bin_extended[- word_len:]))
        result += index_bin_final
    return result

UPPER_BOUND = 0.6
LOWER_BOUND = 0.4
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
        # TODO: continue here (evaluation -> MSE, accuracy on same data)
        pred = model.predict(x, verbose=0)
        avg_mse = mse(y, pred)
        #rel = map(lambda row: , x.iterrows())
        print(f'Epoch {i_step * step}/{episodes}: accuracy -> {avg_mse * 100: .3f}%')
        #print(f'Epoch {i_step * step}/{episodes}: accuracy -> {avg_mse * 100: .3f}%', file=stderr)
        #print(f'Epoch {i_step * step}/{episodes}: accuracy -> {avg_mse * 100: .3f}%\treliability: {: .3f}')

def main() -> None:
    # DONE: create DataFrame
    words = [
        'hello',
        'begin',
        'today',
        'after',
        ' '
    ]
    word_len = len(words[0])
    results = [
        [ 1 if word == words[i] else 0 for word in filter(lambda x: x != ' ', words) ] if words[i] != ' ' else [0] * (word_len - 1)
            for i in range(len(words))
    ]
    data = list()
    for word in words:
        data.append( transform(word, word_len) )
    for i, dat in enumerate(data):
        print(f'"{words[i]}" -> {dat} = {len(dat)}')
        assert len(dat) == word_len * 5
    
    df = pd.DataFrame(data)
    res_df = pd.DataFrame(results)
    print('Created input data')

    print(words)
    print(data)
    print(results)

    # DONE: create model
    model = Sequential()
    model.add(
        layers.Dense(20, input_shape=(5 * word_len,), activation='sigmoid')
    )
    model.add(
        layers.Dense(4, activation='sigmoid')
    )
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print('Model has been created')

    # DONE: choose order of training patterns
    # sequential

    # DONE: train model
    train_with_steps(model, df, res_df, 1_000, 100, shuffle=False, verbose=0, workers=4)
    #model.fit(x=df, y=res_df, epochs=1_000, shuffle=False, verbose=1)

    # DONE: discuss results
    predictions = model.predict(df, verbose=0)
    print()
    pairs = [ (w, p) for w, p in zip(words, predictions) ]
    print('Printing results of prediction using format: [input] -> [prediction] (mapped) ...')
    #success_counter = 0
    for word, pred in pairs:
        #dat_list = dat[1].tolist()
        pred_mapped = map_to_discreet(pred)
        #success_counter += int( pred_mapped == word )
        #print(f'"{word}" \t --> {pred} (mapped)\t {pred_mapped}')
        print(f'"{word}" \t --> [{ " ".join(map(lambda x: f"{x:.3f}",pred))}] (mapped)\t {pred_mapped}')
    #print(f'Success rate: {success_counter}/{len(pairs)}')

if __name__ == '__main__':
    main()