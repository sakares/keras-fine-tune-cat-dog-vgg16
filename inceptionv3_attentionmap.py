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

def get_cat_dog_label(pred_class):
    if pred_class==1:
        return 'dog'
    elif pred_class==0:
        return 'cat'
        

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

image_paths = [
    "data/test/1.jpg", "data/test/2.jpg", "data/test/3.jpg", "data/test/4.jpg", 
    "data/test/5.jpg", "data/test/6.jpg", "data/test/7.jpg", "data/test/8.jpg", 
]

for path in image_paths:
    seed_img = utils.load_img(path, target_size=(299, 299))
    
    # Convert to BGR, create input with batch_size: 1, and predict.
    bgr_img = utils.bgr2rgb(seed_img)
    img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
    pred_class = np.argmax(model.predict(img_input))
    
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
    plt.axis('off')
    # plt.imshow(seed_img)
    plt.imshow(heatmap)
    plt.title('Attention - {}'.format(get_cat_dog_label(pred_class)))
    plt.show()


