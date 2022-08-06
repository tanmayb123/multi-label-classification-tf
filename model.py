import tensorflow as tf
import numpy as np

def get_preprocess_fn():
    crop_layer = tf.keras.layers.CenterCrop(224, 224)
    norm_layer = tf.keras.layers.Normalization(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
    )

    def preprocess_image(image, size):
        image = np.array(image)
        image_resized = tf.expand_dims(image, 0)

        if size == 224:
            image_resized = tf.image.resize(image_resized, (256, 256), method="bicubic")
            image_resized = crop_layer(image_resized)
        elif size == 384:
            image_resized = tf.image.resize(image, (size, size), method="bicubic")

        return norm_layer(image_resized).numpy()

    return preprocess_image

def get_model():
    model = tf.keras.models.load_model("convnext_xlarge_21k_224_fe")
    model.summary()

    return model

if __name__ == "__main__":
    get_model()
