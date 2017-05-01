import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import model_from_json, Model
from vis.losses import ActivationMaximization
from vis.optimizer import Optimizer
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.utils.inception_v3 import InceptionV3, conv2d_bn
from vis.visualization import visualize_saliency, visualize_cam, visualize_activation, get_num_filters
from matplotlib import pyplot as plt

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

#  Build the VGG16 network with ImageNet weights
# model_vgg = VGG16(weights='imagenet', include_top=True)
# model= VGG16(weights='imagenet', include_top=False)
# print('Model loaded.')

# Build the InceptionV3 network with ImageNet weights
# model_inception_v3 = InceptionV3(weights='imagenet', include_top=True)
# model = InceptionV3(weights='vader-yoda', include_top=True)
# print('Model loaded.')

# Custom InceptionV3 network
# load json and create model
# json_file = open('vgg16_tf_cat_dog_final_dense2.json', 'r')
json_file = open('inception_v3_tf_cat_dog_top_layer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# loaded_model.load_weights("vgg16_tf_cat_dog_final_dense2.h5")
loaded_model.load_weights("inception_v3_tf_cat_dog_top_layer.h5")
print("Loaded model from disk")

model_inception = loaded_model

# The name of the layer we want to visualize
# (see model definition in vggnet.py or inception_v3.py)
# layer_name = 'predictions'
layer_name = 'predictions'
# layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
layer_idx = [idx for idx, layer in enumerate(model_inception.layers) if layer.name == layer_name][0]

# vgg_layer_name = 'predictions'
# vgg_layer_idx = [idx for idx, layer in enumerate(model_vgg.layers) if layer.name == vgg_layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
# path = '/data/test'
image_paths = [
    "data/test/1.jpg", "data/test/2.jpg", "data/test/3.jpg", "data/test/4.jpg", 
    "data/test/5.jpg", "data/test/6.jpg", "data/test/7.jpg", "data/test/8.jpg", 
]

heatmaps = []
vis_images = []

# def pred_class_eval(score):
#     if score > 0.5:
#         return np.int64(1) # 'Dog'
#     else:
#         return np.int64(0) # 'Cat'
        
def get_cat_dog_label(pred_class):
    if pred_class==1:
        return 'dog'
    elif pred_class==0:
        return 'cat'
        
path = image_paths[3]
# seed_img = utils.load_img(path, target_size=(299, 299))

# pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))

# seed_img = utils.load_img(path, target_size=(224, 224))
seed_img = utils.load_img(path, target_size=(299, 299))

# Convert to BGR, create input with batch_size: 1, and predict.
bgr_img = utils.bgr2rgb(seed_img)
img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
pred_class = np.argmax(model_inception.predict(img_input))

heatmap = visualize_cam(model_inception, layer_idx, [pred_class], seed_img)
plt.axis('off')
plt.imshow(heatmap)
plt.title('Attention - {}'.format(get_cat_dog_label(pred_class)))
plt.show()

pred_class = np.int64(0)

heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
plt.axis('off')
plt.imshow(heatmap)
plt.title('Attention - {}'.format(get_cat_dog_label(pred_class)))
plt.show()


# pred_otherwise = np.argmin(model.predict(np.array([img_to_array(seed_img)])))
# pred_class_vgg = np.argmax(model_vgg.predict(np.array([img_to_array(seed_img)])))

# heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img, text=get_cat_dog_label(pred_class))
# # heatmap_otherwise = visualize_cam(model, layer_idx, [pred_otherwise], seed_img, text="otherwise")
# # heatmap_vgg = visualize_cam(model_vgg, vgg_layer_idx, [pred_class_vgg], seed_img, text=utils.get_imagenet_label(pred_class_vgg))
# heatmaps.append(heatmap)
# heatmaps.append(heatmap_otherwise)
# heatmaps.append(heatmap_vgg)

# Generate three different images of the same output index.
# vis_images = [visualize_activation(model, layer_idx, filter_indices=idx, text=str(idx), max_iter=500) for idx in [1, 1, 1]]
# vis_images.append(vis_image)

for path in image_paths:
    
    # For InceptionV3
    # seed_img = utils.load_img(path, target_size=(299, 299))
    # pred = model.predict(preprocess_input(np.expand_dims(img_to_array(seed_img), axis=0)))
    
    # For VGG16
    seed_img = utils.load_img(path, target_size=(224, 224))
    # pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
    pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
    # print(pred_class)
    print("***********************************")
    print("custom:")
    print(pred_class)
    print(get_cat_dog_label(pred_class))
    
    
    # seed_img = utils.load_img(path, target_size=(224, 224))
    # pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
    # print(pred)
    
    # print('Predicted:', decode_predictions(pred))
    # print('Predicted:', decode_predictions(pred)[0][0][1])
    
    # print('Predicted:', decode_predictions_vader_yoda(pred))
    # print('Predicted:', decode_predictions_vader_yoda(pred)[0][0][1])
    
    
    # pred_class_vgg = np.argmax(model_vgg.predict(preprocess_input(np.array([img_to_array(seed_img)]))))
    pred_class_vgg = np.argmax(model_vgg.predict(np.array([img_to_array(seed_img)])))
    print("vgg16:")
    print(pred_class_vgg)
    print(utils.get_imagenet_label(pred_class_vgg))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    # visualize_saliency
    # heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img, text=("Custom: %s" % get_cat_dog_label(pred_class)))
    # heatmap_vgg = visualize_saliency(model_vgg, vgg_layer_idx, [pred_class_vgg], seed_img, text=("VGG16: %s" % utils.get_imagenet_label(pred_class_vgg)))
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img, text=("Custom: %s" % get_cat_dog_label(pred_class)))
    heatmap_vgg = visualize_cam(model_vgg, vgg_layer_idx, [pred_class_vgg], seed_img, text=("VGG16: %s" % utils.get_imagenet_label(pred_class_vgg)))
    heatmaps.append(heatmap)
    heatmaps.append(heatmap_vgg)
    
    # Generate three different images of the same output index.
    # vis_images = [visualize_activation(model, layer_idx, filter_indices=idx, text=str(idx), max_iter=500) for idx in [294, 294, 294]]
    # vis_images.append(vis_image)

name = "Gradient-based Localization map"
cv2.imshow(name, utils.stitch_images(heatmaps))
cv2.waitKey(-1)
cv2.destroyWindow(name)


# name = "Visualizations » Dense Layers"
# cv2.imshow(name, utils.stitch_images(vis_images))
# cv2.waitKey(-1)
# cv2.destroyWindow(name)



