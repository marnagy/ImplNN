import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses

import pandas as pd
import numpy as np
from pprint import pprint
from itertools import chain
from tqdm import tqdm

LETTERS = 'qwertyuiopasdfghjklzxcvbnm'
POSITIVE_COEF = 300
NEGATIVE_COEF = 700

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

def get_neg_words(amount: int, pos_words: list[str]) -> list[str]:
    res = list()
    for _ in range(amount):
        word = np.random.choice(pos_words)
        word_len = len(word)
        index = np.random.randint(0, word_len)
        new_letter = np.random.choice(list(LETTERS))
        res_word = word[:index] + new_letter + word[index+1:]
        res.append( res_word )
    return res

def train_with_steps(model: Sequential, x: pd.DataFrame, y: pd.DataFrame, episodes: int, step: int, **kwargs) -> None:
    if episodes % step != 0:
        raise Exception("Episodes must be divisible by step")

    mse = losses.MeanSquaredError()

    steps_count = episodes // step
    for i_step in tqdm(range(steps_count), ascii=True, total=steps_count):
        model.fit(x, y, epochs=step,  **kwargs) #initial_epoch=i_step * step,
        # TODO: continue here (evaluation -> MSE, accuracy on same data)
        pred = model.predict(x, verbose=0)
        avg_mse = mse(y, pred)
        #rel = map(lambda row: , x.iterrows())
        print(f'Epoch {i_step * step}/{episodes}: accuracy -> {avg_mse * 100: .3f}%')
        #print(f'Epoch {i_step * step}/{episodes}: accuracy -> {avg_mse * 100: .3f}%', file=stderr)
        #print(f'Epoch {i_step * step}/{episodes}: accuracy -> {avg_mse * 100: .3f}%\treliability: {: .3f}')

def main() -> None:
    np.random.seed = 42

    # DONE: create DataFrame
    words = [
        'hello',
        'begin',
        'today',
        'after',
        ' '
    ]
    neg_words = get_neg_words(NEGATIVE_COEF, words[:-1])
    #print(f'Negative words: {neg_words}')
    word_len = len(words[0])

    pos_results = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,0]
    ]

    words_input = words[:]

    words = POSITIVE_COEF * words

    pos_results = POSITIVE_COEF * pos_results
    neg_results = len(neg_words) * [len(pos_results[0]) * [0]]
    results = pos_results + neg_results

    pos_data = list()
    for word in words:
        pos_data.append( transform(word, word_len) )
    neg_data = list()
    for neg_word in neg_words:
        neg_data.append( transform(neg_word, word_len) )
    
    data = pos_data + neg_data

    for i, dat in enumerate(data):
        #print(f'{dat} => {results[i]} - {len(dat)}')
        assert len(dat) == word_len * 5
    
    df = pd.DataFrame(data)
    res_df = pd.DataFrame(results)
    print('Created input data')

    words = words + neg_words

    #pprint(len(words))
    #pprint(len(data))
    #pprint(len(results))

    # print(f'data shape: {}')

    # DONE: create model
    input_shape = (5 * word_len,)
    model = Sequential()
    model.add(
        layers.Dense(20, input_shape=input_shape, activation='sigmoid')
    )
    model.add(
        layers.Dense(4, activation='sigmoid')
    )
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print('Model has been created')

    # DONE: choose order of training patterns
    # sequential

    # DONE: train model
    train_with_steps(model, df, res_df, 5_000, 100, shuffle=False, verbose=0, workers=4, use_multiprocessing=True, batch_size=64)
    #model.fit(x=df, y=res_df, epochs=1_000, shuffle=False, verbose=1)

    # DONE: discuss results
    test_words = words_input + get_neg_words(20, words_input[:-1])
    test_df = pd.DataFrame(list(map(lambda x: transform(x, word_len), test_words)))
    predictions = model.predict(test_df, verbose=0)
    print()
    pairs = [ (w, p) for w, p in zip(test_words, predictions) ]
    print('Printing results of prediction using format: [input] -> [prediction] (mapped) ...')
    #success_counter = 0
    for word, pred in pairs:
        #dat_list = dat[1].tolist()
        pred_mapped = map_to_discreet(pred)
        #success_counter += int( pred_mapped == word )
        #print(f'"{word}" \t --> {pred} (mapped)\t {pred_mapped}')
        print(f'"{word}" \t --> [{ " ".join(map(lambda x: f"{x:.3f}",pred))}] (mapped)\t {pred_mapped}')
    #print(f'Success rate: {success_counter}/{len(pairs)}')

    #print(f'words_input: {words_input}')

if __name__ == '__main__':
    main()