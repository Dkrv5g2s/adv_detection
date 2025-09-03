import tensorflow as tf
import keras
import numpy as np
from keras import layers
from keras.utils import to_categorical
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 禁用 Eager Execution
tf.compat.v1.disable_eager_execution()

# 限制 GPU 記憶體使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)

# 模型和數據參數
num_classes = 10
input_shape = (28, 28, 1)

# 加載 MNIST 數據集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 數據預處理
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 模型構建
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

# 使用 tf.compat.v1.Session 訓練模型
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    classifier = KerasClassifier(model=model, clip_values=(0, 1), sess=sess)

    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.1)

    # ART FGSM 攻擊
    attack = FastGradientMethod(estimator=classifier, eps=0.1)
    print("Generating FGSM samples...")
    x_train_adv = attack.generate(x=x_train[:100])  # 減少數據大小
    print("FGSM attack samples generated successfully.")

    # SHAP 值生成
    logit_layer_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    background = x_train[np.random.choice(x_train.shape[0], 10, replace=False)]
    explainer = shap.DeepExplainer(logit_layer_model, background)

    def compute_shap_values(explainer, data, batch_size=5):
        shap_values = []
        for i in range(0, len(data), batch_size):
            shap_values.extend(explainer.shap_values(data[i:i + batch_size]))
        return np.array(shap_values)

    print("Computing SHAP values...")
    shap_values_original = compute_shap_values(explainer, x_train[:50], batch_size=5)
    shap_values_adv = compute_shap_values(explainer, x_train_adv[:50], batch_size=5)
    print("SHAP values computed successfully.")

    # 特徵生成
    def extract_shap_signature(shap_values):
        return np.array([sv.flatten() for sv in shap_values])

    shap_signature_original = extract_shap_signature(shap_values_original)
    shap_signature_adv = extract_shap_signature(shap_values_adv)

    # 訓練檢測器
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

    # 評估檢測器
    y_pred = np.argmax(detector_model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred))