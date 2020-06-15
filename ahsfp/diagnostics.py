from keras import backend as K
import numpy as np
from scipy.interpolate import interp2d
import cv2

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

        pooled_grads_value = np.ones(pooled_grads_value.shape) if \
            np.sum(pooled_grads_value) == 0.0 else pooled_grads_value

        pooled_grads_value /= np.sum(pooled_grads_value)

        heatmap = np.abs(np.average(conv_layer_output_value, weights=pooled_grads_value, axis=2))
        heatmap[heatmap < tol] = 0
        return heatmap

    def activation_map(self, x, tol=1e-3):

        img_tensor = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        hm = self.generate_heatmap(img_tensor=img_tensor, tol=tol)

        hm = (hm - hm.min()) / (hm.max() - hm.min())

        x_org = np.linspace(0, 1, len(hm))
        f = interp2d(x_org, x_org, hm)

        x_new = np.linspace(0, 1, len(x[0]))
        return f(x_new, x_new)

    def draw_activation(self, x, tol=1e-3, alpha=0.5):

        image = Image.fromarray((255 - x[:, :, 0] * 255).astype('uint8'))
        image.save('temp.png')

        image = cv2.imread('temp.png')

        heatmap = self.activation_map(x, tol=tol)
        colormap = cv2.COLORMAP_VIRIDIS

        heatmap = (heatmap * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, colormap)
        return cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # maxlen = 130
        # smi = Smile()
        # smiles_x = smi.smile_to_sequence(smiles)
        # smiles_x = sequence.pad_sequences([smiles_x], maxlen=maxlen)
        #
        # img_tensor = smiles_x[0].reshape(1, maxlen, 1)
        # heatmap, heatmap_org = self.generate_heatmap(img_tensor, class_index=class_index,
        #                                              label_index=label_index, tol=tol, norm=norm)
        # if len(heatmap) < 2:
        #     return None, None, None
        # x_org = np.linspace(0, 1, len(heatmap))
        # x_new = np.linspace(0, 1, len(smiles))
        # f1 = interp1d(x_org, heatmap, kind='nearest')
        #
        # hm = f1(x_new)
        # hm = hm.reshape(hm.size)
        #
        # hm = (hm - hm.min()) / (hm.max() - hm.min())
        #
        # fact = float(len(heatmap)) / float(len(smiles))
        #
        # return fact, hm, heatmap_org

