import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses

import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps, ImageChops, ImageTransform
from tqdm import tqdm
from argparse import Namespace, ArgumentParser

INPUTS_DIR = 'inputs_linart'
INPUT_FILES = ['gulocka', 'lomka_l', 'lomka_p', 'podciarka', 'pomlcka', 'vokan', 'zvyslitko']
BLANK_FILE = 'base.png'
INPUT_FILES = list(map(lambda x: x + '.png', INPUT_FILES))

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--load', action='store_true')
    parser.set_defaults(load=False)

    parser.add_argument('--epochs', default=5_000, type=int)
    
    parser.add_argument('--model_name', default='linart_model_2x64.h5', type=str)
    parser.add_argument('--test_img', default='diamantoid_bb.png', type=str)

    return parser.parse_args()

def duplicate(arr, n) -> list:
    return [ele for ele in arr for _ in range(n)]

def train_with_steps(model: Sequential, x: pd.DataFrame, y: pd.DataFrame, episodes: int, step: int, **kwargs) -> None:
    if episodes % step != 0:
        raise Exception("Episodes must be divisible by step")

    #mse = losses.MeanSquaredError()
    categorical_entropy = losses.CategoricalCrossentropy()

    steps_count = episodes // step
    for i_step in range(steps_count):
        model.fit(x, y, epochs=step,  **kwargs) #initial_epoch=i_step * step,
        # TODO: continue here (evaluation -> MSE, accuracy on same data)
        pred = model.predict(x, verbose=0)
        cat_entropy = categorical_entropy(y, pred) * 100
        #rel = map(lambda row: , x.iterrows())
        upd_str = f'{(i_step + 1) * step}/{episodes}: MSE -> {cat_entropy: .2f}'
        print(upd_str)

def one_hot(index: int, length: int) -> np.ndarray:
    a = np.zeros(length, 'float64')
    a[index] = 1
    return a

def main():
    np.random.seed = 42
    tf.random.set_seed(42)

    args = get_args()

    # DONE: load images
    images = list( map(lambda file: Image.open(os.path.join(INPUTS_DIR, file)).convert('1'), INPUT_FILES) )
    possible_results = [' ', 'o', '\\', '/', '_', '-', '^', '|']
    correct_classification = possible_results[:][1:]

    # DONE: create multiple transformations (rotate, move)
    rotations = [ 0, 2, -2, 4, -4 ] # in degrees
    moves_x   = [ 0, 1, -1, 2, -2, 3, -3 ] # in px
    moves_y   = [ 0, 1, -1, 2, -2, 3, -3, 4, -4 ] # in px
    transformed_images: list[Image.Image] = list()
    for img in images:
        for rot in rotations:
            for mov_x in moves_x:
                for mov_y in moves_y:
                    new_img = img
                    if rot != 0:
                        new_img = new_img.rotate(rot, fillcolor=1)

                    # TODO: skip moves in X axis for podciarka

                    new_img = new_img.transform(img.size, Image.Transform.AFFINE, (1, 0, mov_x, 0, 1, mov_y), fill=1, fillcolor=1)
                    
                    transformed_images.append( np.array(new_img) )
    del images

    correct_classification = duplicate(correct_classification, len(rotations) * len(moves_x) * len(moves_y))

    # add blank img
    blank_img = np.array( Image.open(os.path.join(INPUTS_DIR, BLANK_FILE)).convert('1') )
    for _ in range( 200 ):
        transformed_images.append( blank_img )
        correct_classification.append(' ')
    
    print(f'Total training images: { len(transformed_images) }')

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
    input_shape = (12,8)
    if args.load:
        del transformed_images
        model = tf.keras.models.load_model(args.model_name)
    else:
        model = Sequential()
        model.add(
            tf.keras.Input(shape=input_shape)
        )
        model.add(
            layers.Flatten()
        )

        for _ in range(3):
            model.add(
                layers.Dense(64, activation='sigmoid')
            )

        # model.add(
        #     layers.Dense(64, activation='sigmoid')
        # )
        # model.add(
        #     layers.Dense(32, activation='sigmoid')
        # )
        # model.add(
        #     layers.Dense(16, activation='sigmoid')
        # )

        model.add(
            layers.Dense(8, activation='sigmoid')
        )
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        print('Model has been created.')

        # TODO: train model
        # !: error on 'train'
        train_input_df = np.array(transformed_images, dtype=np.float64)
        del transformed_images
        #results = np.expand_dims(np.array([ tf.one_hot(possible_results.index(res), len(possible_results)) for res in correct_classification ]), axis=1)
        results = np.array([ tf.one_hot(possible_results.index(res), len(possible_results)) for res in correct_classification ])
        # print(train_input_df.shape, train_input_df.dtype)
        # print(results.shape, results.dtype)

        #model.fit(train_input_df, results, epochs=1_000)

        try:
            train_with_steps(model, train_input_df, results, args.epochs, 100,
                shuffle=True, verbose=0, workers=4, use_multiprocessing=False, batch_size=32)
        except KeyboardInterrupt:
            pass

        if input(f'Overwrite model as {args.model_name} ? ').lower() == 'y':
            print(f'Saving model to {args.model_name}')
            model.save(args.model_name)        

    model: Sequential

    # TODO: show result on bigger picture
    test_img = Image.open(
        os.path.join(INPUTS_DIR, args.test_img)
    ).convert('1')
    test_input = np.array(test_img, dtype=np.float64)

    for i_x in range(test_input.shape[0] // input_shape[0]):
        row = test_input[input_shape[0] * i_x: input_shape[0] * (i_x + 1)]
        for i_y in range(test_input.shape[1] // input_shape[1]):

            cropped_input = row[:, input_shape[1] * i_y: input_shape[1] * (i_y + 1)]
            #print(cropped_input)
            pred = model.predict( np.expand_dims(cropped_input, axis=0), verbose=0)
            pred_index = np.argmax(pred)
            print(f'{possible_results[pred_index]}', end='')
        print()

if __name__ == '__main__':
    main()