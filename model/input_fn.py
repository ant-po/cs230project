"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np

#
def _parse_function(data_index, label, alldata):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    data = tf.gather(tf.constant(alldata), data_index)
    return data, label
#
#
# def train_preprocess(image, label, use_random_flip):
#     """Image preprocessing for training.
#
#     Apply the following operations:
#         - Horizontally flip the image with probability 1/2
#         - Apply random brightness and saturation
#     """
#     if use_random_flip:
#         image = tf.image.random_flip_left_right(image)
#
#     image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
#     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
#
#     # Make sure the image is still in [0, 1]
#     image = tf.clip_by_value(image, 0.0, 1.0)
#
#     return image, label


def input_fn(is_training, alldata, labels, params):
    """Input function for the SIGNS dataset.
    TODO: update description here
    The data has format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        data: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = alldata.shape[0]
    assert num_samples == labels.shape[0], "Data and labels should have same length"
    data_index = np.arange(num_samples)
    # Create a Dataset serving batches of data and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda d, l: _parse_function(d, l, alldata)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_index, shape=[num_samples, 1]), tf.constant(labels, shape=[num_samples, 1])))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_index), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    data, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'data': data, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
