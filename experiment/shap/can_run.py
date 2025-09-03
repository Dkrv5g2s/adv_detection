import keras
import numpy as np
from keras import layers
from keras.utils import to_categorical
import shap
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 限制 TensorFlow 記憶體使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# **1. 模型和數據參數**
num_classes = 10
input_shape = (28, 28, 1)

# **2. 加載 MNIST 數據集**
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# **3. 數據預處理**
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# **4. 模型構建**
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
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

# **解決問題：初始化模型輸入屬性**
# 在模型構建完成後進行一次前向傳播
_ = model.predict(x_train[:1])

# **5. FGSM 攻擊樣本生成**
def generate_fgsm_samples(model, x_data, y_data, epsilon=0.1):
    x_adv = x_data.copy()
    y_true = tf.convert_to_tensor(y_data, dtype=tf.float32)
    x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor)
        loss = tf.keras.losses.categorical_crossentropy(y_true, predictions)

    gradients = tape.gradient(loss, x_tensor)
    x_adv += epsilon * tf.sign(gradients)
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv.numpy()

x_train_adv = generate_fgsm_samples(model, x_train, y_train, epsilon=0.1)

# **6. SHAP 值生成**
# 使用 model.inputs 替代 model.input
logit_layer_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

# 減少背景數據大小
background = x_train[np.random.choice(x_train.shape[0], 20, replace=False)]

explainer = shap.DeepExplainer(logit_layer_model, background)

# 分批計算 SHAP 值
def compute_shap_values(explainer, data, batch_size=10):
    shap_values = []
    for i in range(0, len(data), batch_size):
        shap_values.extend(explainer.shap_values(data[i:i + batch_size]))
    return np.array(shap_values)

shap_values_original = compute_shap_values(explainer, x_train[:100])
print(shap_values_original.shape)
shap_values_adv = compute_shap_values(explainer, x_train_adv[:100])

# **7. 特徵生成：SHAP 簽名**
def extract_shap_signature(shap_values):
    return np.array([sv.flatten() for sv in shap_values])

shap_signature_original = extract_shap_signature(shap_values_original)
print(shap_signature_original.shape)
shap_signature_adv = extract_shap_signature(shap_values_adv)

# **8. 訓練檢測器（DNN 模型：256-128-16-2）**
X = np.vstack([shap_signature_original, shap_signature_adv])
y = np.array([0] * len(shap_signature_original) + [1] * len(shap_signature_adv))

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
detector_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# **9. 評估檢測器**
y_pred = np.argmax(detector_model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))