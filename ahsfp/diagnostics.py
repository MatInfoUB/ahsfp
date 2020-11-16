from keras import backend as K
import numpy as np
from scipy.interpolate import interp2d
import cv2
from PIL import Image
from keras.wrappers import scikit_learn as sklearn

class Activation:

    def __init__(self, model, layer_ind=-1):

        # Instatiate with a classifier
        self.model = model.model

        # Find the last conv layer
        conv_layers = [layer.name for layer in self.model.layers if 'conv' in layer.name]
        self.last_conv_layer = self.model.get_layer(conv_layers[layer_ind])

    def compute_gradient(self):

        output = self.model.output
        grads = K.gradients(output, self.last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        return K.function([self.model.input], [pooled_grads, self.last_conv_layer.output[0]])

    def generate_heatmap(self, img_tensor, tol=1e-3):

        iterate = self.compute_gradient()
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
        pooled_grads_value = np.abs(pooled_grads_value)
        # pooled_grads_value = np.ones(pooled_grads_value.shape) if \
        #     np.sum(pooled_grads_value) == 0.0 else pooled_grads_value

        pooled_grads_value /= np.sum(pooled_grads_value)

        heatmap = np.abs(np.average(conv_layer_output_value, weights=pooled_grads_value, axis=2))
        heatmap[heatmap < tol] = 0
        return heatmap

    def activation_map(self, x, tol=1e-3):

        img_tensor = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        hm = self.generate_heatmap(img_tensor=img_tensor, tol=tol)

        hm = (hm - hm.min()) / (hm.max() - hm.min())

        permutation = list(range(4, len(hm))) + list(range(0, 4))
        hm = hm[:, permutation]

        x_org = np.linspace(0, 1, len(hm))
        f = interp2d(x_org, x_org, hm)

        x_new = np.linspace(0, 1, len(x[0]))
        return f(x_new, x_new)

    def draw_activation(self, x, tol=1e-3, alpha=0.5):

        image = Image.fromarray((255 - x[:, :, 0] * 255).astype('uint8'))
        image.save('temp.png')

        image = cv2.imread('temp.png')

        hm_original = self.activation_map(x, tol=tol)
        colormap = cv2.COLORMAP_VIRIDIS

        heatmap = (hm_original * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, colormap)
        return cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0), hm_original


class Saliency:

    def __init__(self, model=None):

        self.model = model.model

    def saliency_map(self, x, tol=1e-3):

        output = self.model.output
        grads = K.gradients(output, self.model.input)[0][0]
        iterate = K.function([self.model.input], [grads])

        grads_value = iterate([x[np.newaxis, :, :, :]])[0]
        grads_value = grads_value[:, :, 0]
        grads_value[np.abs(grads_value) < tol] = 0

        g_plus = grads_value.copy()
        g_neg = grads_value.copy()

        g_neg[g_neg > 0] = 0
        g_neg = np.abs(g_neg)

        g_plus[g_plus < 0] = 0
        # tol = np.square(np.mean(grads_value ** 2))
        # grads_value /= tol

        return grads_value, g_plus, g_neg

    def draw_activation(self, x, tol=1e-3, alpha=0.5):

        image = Image.fromarray((255 - x[:, :, 0] * 255).astype('uint8'))
        image.save('temp.png')

        image = cv2.imread('temp.png')

        hm_original, _, _ = self.saliency_map(x)
        colormap = cv2.COLORMAP_VIRIDIS

        heatmap = (hm_original * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, colormap)
        return cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0), hm_original

class GradCAMplus:

    def __init__(self, model, layer_ind=-1):

        # Instatiate with a classifier
        self.model = model.model

        # Find the last conv layer
        conv_layers = [layer.name for layer in self.model.layers if 'conv' in layer.name]
        self.last_conv_layer = self.model.get_layer(conv_layers[layer_ind])

    def compute_gradient(self):

        output = self.model.output
        grads = K.gradients(output, self.last_conv_layer.output)[0]
        return K.function([self.model.input], [grads, self.last_conv_layer.output[0]])

    def generate_heatmap(self, img_tensor, tol=1e-3):
        iterate = self.compute_gradient()
        grads_value, conv_layer_output_value = iterate([img_tensor])
        grads_value = grads_value[0]

        hm_original = np.zeros(grads_value.shape[0:2])
        sum = 0
        for k in range(grads_value.shape[-1]):
            pooled_grads_value = 0
            Ak = conv_layer_output_value[:, :, k]
            ind = Ak > 0
            Ak_plus = Ak * ind
            if ind.any():
                pooled_grads_value = Ak_plus.sum() / ind.sum()
            sum += pooled_grads_value
            hm_original += pooled_grads_value * Ak

        hm_original /= sum
        hm_original[hm_original < tol] = 0
        return hm_original

    def activation_map(self, x, tol=1e-3):

        img_tensor = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        hm = self.generate_heatmap(img_tensor=img_tensor, tol=tol)

        hm = (hm - hm.min()) / (hm.max() - hm.min())
        permutation = list(range(2, len(hm))) + list(range(0, 2))
        hm = hm[:, permutation]

        x_org = np.linspace(0, 1, len(hm))
        f = interp2d(x_org, x_org, hm)

        x_new = np.linspace(0, 1, len(x[0]))
        return f(x_new, x_new)

    def draw_activation(self, x, tol=1e-3, alpha=0.5):

        image = Image.fromarray((255 - x[:, :, 0] * 255).astype('uint8'))
        image.save('temp.png')

        image = cv2.imread('temp.png')

        hm_original = self.activation_map(x, tol=tol)
        colormap = cv2.COLORMAP_VIRIDIS

        heatmap = (hm_original * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, colormap)
        return cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0), hm_original