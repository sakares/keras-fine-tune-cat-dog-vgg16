import numpy as np

from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import model_from_json, Model
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.utils.inception_v3 import InceptionV3
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam


def generate_saliceny_map(show=True):
    """Generates a heatmap indicating the pixels that contributed the most towards
    maximizing the filter output. First, the class prediction is determined, then we generate heatmap
    to visualize that class.
    """
    # # Build the VGG16 network with ImageNet weights
    # model = VGG16(weights='imagenet', include_top=True)
    # print('Model loaded.')
    
    # Build the InceptionV3 network with ImageNet weights
    model = InceptionV3(weights='imagenet', include_top=True)
    print('Model loaded.')
    print(model.summary())

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['../resources/ouzel.jpg', '../resources/ouzel_1.jpg']:
        # seed_img = utils.load_img(path, target_size=(224, 224))
        seed_img = utils.load_img(path, target_size=(299, 299))

        # Convert to BGR, create input with batch_size: 1, and predict.
        bgr_img = utils.bgr2rgb(seed_img)
        img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
        pred_class = np.argmax(model.predict(img_input))

        heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
        if show:
            plt.axis('off')
            plt.imshow(heatmap)
            plt.title('Saliency - {}'.format(utils.get_imagenet_label(pred_class)))
            plt.show()


def generate_cam(show=True):
    """Generates a heatmap via grad-CAM method.
    First, the class prediction is determined, then we generate heatmap to visualize that class.
    """
    # # Build the VGG16 network with ImageNet weights
    # model = VGG16(weights='imagenet', include_top=True)
    # print('Model loaded.')
    
    # Build the InceptionV3 network with ImageNet weights
    # model = InceptionV3(weights='imagenet', include_top=True)
    # print('Model loaded.')
    
    json_file = open('inception_v3_tf_cat_dog_top_layer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # loaded_model.load_weights("vgg16_tf_cat_dog_final_dense2.h5")
    loaded_model.load_weights("inception_v3_tf_cat_dog_top_layer.h5")
    print("Loaded model from disk")

    model = loaded_model


    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['../resources/tiger.jpg','../resources/ouzel.jpg', '../resources/ouzel_1.jpg']:
        # seed_img = utils.load_img(path, target_size=(224, 224))
        seed_img = utils.load_img(path, target_size=(299, 299))

        # Convert to BGR, create input with batch_size: 1, and predict.
        bgr_img = utils.bgr2rgb(seed_img)
        img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
        pred_class = np.argmax(model.predict(img_input))

        heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
        if show:
            plt.axis('off')
            plt.imshow(heatmap)
            plt.title('Attention - {}'.format(utils.get_imagenet_label(pred_class)))
            plt.show()


if __name__ == '__main__':
    generate_cam()
    generate_saliceny_map()

