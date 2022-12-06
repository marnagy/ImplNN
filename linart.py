import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses

import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps, ImageChops
from tqdm import tqdm

INPUTS_DIR = 'inputs_linart'
INPUT_FILES = ['gulocka', 'lomka_l', 'lomka_p', 'podciarka', 'pomlcka', 'vokan', 'zvyslitko']
BLANK_FILE = 'base.png'
INPUT_FILES = list(map(lambda x: x + '.png', INPUT_FILES))

def duplicate(arr, n) -> list:
    return [ele for ele in arr for _ in range(n)]

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

def one_hot(index: int, length: int) -> np.ndarray:
    a = np.zeros(length, 'int8')
    a[index] = 1
    return a

def main():
    np.random.seed = 42

    # DONE: load images
    images = list( map(lambda file: Image.open(os.path.join(INPUTS_DIR, file)).convert('1'), tqdm(INPUT_FILES, ascii=True)) )
    possible_results = [' ', 'o', '\\', '/', '_', '-', '^', '|']
    correct_classification = possible_results[:][1:]

    # DONE: create multiple transformations (rotate, move)
    rotations = [ 0, 3, -3, 7, -7 ] # in degrees
    moves     = [ 0, 1, -1, 2, -2 ] # in px
    transformed_images: list[Image.Image] = list()
    for img in tqdm(images, ascii=True):
        for rot in rotations:
            for mov_x in moves:
                for mov_y in moves:
                    new_img = img
                    if rot != 0:
                        new_img = new_img.rotate(rot, fillcolor=1)

                    new_img = new_img.transform(img.size, Image.AFFINE, (1, 0, mov_x, 0, 1, mov_y), fill=1, fillcolor=1)
                    
                    transformed_images.append( new_img )
    del images

    correct_classification = duplicate(correct_classification, 5**3)

    # add blank img
    for _ in range(5):
        transformed_images.append( Image.open(os.path.join(INPUTS_DIR, BLANK_FILE)).convert('1') )
        correct_classification.append(possible_results[0])

    save_transformed_imgs = False
    if save_transformed_imgs:
        res_dir = 'transformed_imgs'
        try:
            os.makedirs(res_dir, exist_ok=False)

            for i, img in enumerate(transformed_images):
                img.save(os.path.join(res_dir, f'{i}.png'))
        except:
            pass

    # DONE: create model (input 12x8)
    model = Sequential()
    model.add(
       layers.Dense(20, input_shape=(12, 8), activation='sigmoid')
    )
    model.add(
       layers.Dense(8, activation='sigmoid')
    )
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print('Model has been created.')

    # TODO: train model
    # !: error on 'train'
    train_input_df = np.array([ np.array(img) for img in transformed_images ])
    results = np.array([ one_hot(possible_results.index(res), len(possible_results)) for res in correct_classification ])
    print(train_input_df.shape)
    print(results.shape)
    train_with_steps(model, train_input_df, results, 1_000, 100,
        shuffle=False, verbose=0, workers=4, use_multiprocessing=False, batch_size=64)

    # TODO: show result on bigger picture

if __name__ == '__main__':
    main()