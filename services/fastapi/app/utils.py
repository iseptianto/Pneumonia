import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_pil(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB").resize(target_size)
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2"):
    # img_array: (1, H, W, 3)
    # Dapatkan model sampai conv terakhir
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # kelas positif (pneumonia) asumsi sigmoid
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap  # (Hc, Wc)

def overlay_heatmap_on_image(heatmap, orig_pil, alpha=0.35):
    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize(orig_pil.size)
    heatmap = np.array(heatmap)
    heatmap = np.uint8(plt_colormap(heatmap))  # apply colormap
    overlay = Image.blend(orig_pil.convert("RGBA"), heatmap.convert("RGBA"), alpha)
    return overlay

def plt_colormap(gray):
    # convert gray [0..255] to jet-like rgba PIL image
    import matplotlib.cm as cm
    colored = cm.get_cmap("jet")(gray / 255.0)  # (H,W,4) float
    img = (colored * 255).astype("uint8")
    return Image.fromarray(img)
