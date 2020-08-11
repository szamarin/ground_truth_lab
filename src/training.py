import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
import numpy as np
import os
import json
import argparse


def load_and_split(path, label):
    
    """Loads and splits the image and returns a tuple of image pair and label
    
    Parameters
    ----------
    path : str
        The location of the paired image
    label : int,
        Inidcates whether the images are different(0) or same(1) 

    Returns
    -------
    tuple[tuple[Tensor[float]], Tensor[float]]
        a tuple of image pairs and the label
    """

    tf_path = tf.strings.join([f"{os.environ.get('SM_CHANNEL_TRAINING')}/", path])
    
    image = tf.io.read_file(tf_path)
    image = tf.image.decode_png(image, channels=1)
    image /= 255
    image1, image2 = image[:,:92,:], image[:,93:,:]
    
    return (image1, image2), float(label)

def create_data_set(manifest_path, batch_size=32):
    
    "read the manifest file and prep data for training"
    
    with open(manifest_path, "r") as f:
        manifest = f.readlines()
    
    #create an array of image file path and label pairs
    source_label = np.asarray([(os.path.basename(x["source-ref"]), 
                                x["face-labeling"]) for x in map(json.loads, manifest)])
    #convert to tf Dataset
    data = tf.data.Dataset.from_tensor_slices((source_label[:,0], source_label[:,1]))
    #prep data for training
    data = data.map(load_and_split)
    data = data.shuffle(1000).batch(batch_size).repeat()
    
    return data

def create_shared_network(input_shape):
    
    "create the shared component of the Siamese network"
    
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', 
                     input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    return model

@tf.function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


@tf.function
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
                 )

def model(data, epochs, steps_per_epoch):
    
    "Create and train the model"
    
    input_shape = (112, 92, 1)
    shared_network = create_shared_network(input_shape)
    
    input_top = Input(shape=input_shape, name="input_top")
    input_bottom = Input(shape=input_shape, name="input_bottom")
    
    output_top = shared_network(input_top)
    output_bottom = shared_network(input_bottom)
    
    distance = Lambda(euclidean_distance, output_shape=(1,))([output_top, 
                  output_bottom])
    
    model = Model(inputs=[input_top, input_bottom], outputs=distance)
    
    model.compile(loss=contrastive_loss, optimizer="adam")
    
    model.fit(data, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    return model

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    
    manifest_path = os.path.join(os.environ.get('SM_CHANNEL_TRAINING'), "manifests/output/output.manifest")
    
    data = create_data_set(manifest_path, args.batch_size)
    
    model = model(data, args.epochs, args.steps_per_epoch)
    
    model.save(os.path.join(os.environ.get('SM_MODEL_DIR'), '000000001'))
