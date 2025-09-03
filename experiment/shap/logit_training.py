import keras
import numpy as np
from keras import layers
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# **1. 限制 TensorFlow 記憶體使用**
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 限制為 4GB
            )
    except RuntimeError as e:
        print(e)

# **2. 模型和數據參數**
num_classes = 10
input_shape = (28, 28, 1)

# **3. 加載 MNIST 數據集**
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# **4. 數據預處理**
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# **5. 構建分類模型**
model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("開始訓練分類模型")
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)
print("分類模型訓練完成")

# **6. 初始化模型輸入屬性**
_ = model.predict(x_train[:1])


# **7. 分批生成 FGSM 攻擊樣本**
def generate_fgsm_samples_batch(model, x_data, y_data, epsilon=0.1, batch_size=128):
    x_adv = []
    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i:i + batch_size]
        y_batch = y_data[i:i + batch_size]

        x_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        y_true = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = model(x_tensor)
            loss = tf.keras.losses.categorical_crossentropy(y_true, predictions)

        gradients = tape.gradient(loss, x_tensor)
        x_batch_adv = x_batch + epsilon * tf.sign(gradients)
        x_batch_adv = tf.clip_by_value(x_batch_adv, 0, 1)
        x_adv.append(x_batch_adv.numpy())

    return np.vstack(x_adv)


print("開始生成 FGSM 攻擊樣本")
x_train_adv = generate_fgsm_samples_batch(model, x_train, y_train, epsilon=0.1)
print("FGSM 攻擊樣本生成完成")

# **8. 提取 Logit 層輸出**
logit_layer_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)


def extract_logits(model, data):
    return model.predict(data)


print("開始提取 Logit 層輸出")
logits_original = extract_logits(logit_layer_model, x_train[:100])
print(f'logits_original shape:{logits_original.shape}')
logits_adv = extract_logits(logit_layer_model, x_train_adv[:100])
print("Logit 層輸出提取完成")

# **9. 訓練檢測器**
X = np.vstack([logits_original, logits_adv])
y = np.array([0] * len(logits_original) + [1] * len(logits_adv))  # 0: 原始樣本, 1: 攻擊樣本

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

detector_model = keras.Sequential(
    [
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

detector_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("開始訓練檢測器模型")
detector_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)
print("檢測器模型訓練完成")

# **10. 評估檢測器**
print("開始評估檢測器模型")
y_pred = np.argmax(detector_model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))