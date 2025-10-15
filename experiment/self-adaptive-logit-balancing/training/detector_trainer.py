"""
對抗樣本檢測器訓練
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class DetectorTrainer:
    def __init__(self, detector, classifier, device, lr=0.001):
        self.detector = detector
        self.classifier = classifier
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(detector.parameters(), lr=lr)

    def extract_features(self, images, batch_size=100):
        """提取 Log-Softmax 特徵"""
        self.classifier.eval()
        patterns = []

        images_tensor = torch.FloatTensor(images).to(self.device)
        num_batches = (len(images_tensor) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images_tensor))
            batch_images = images_tensor[start_idx:end_idx]

            with torch.no_grad():
                outputs = self.classifier(batch_images)
                log_softmax = F.log_softmax(outputs, dim=1)
                # 排序 log-softmax 值作為特徵
                sorted_log_softmax, _ = torch.sort(log_softmax, dim=1)
                patterns.append(sorted_log_softmax.cpu().numpy())

        return np.vstack(patterns)

    def prepare_training_data(self, adversarial_data):
        """準備訓練數據"""
        print("\n[INFO] Extracting features for detector training...")

        all_patterns = []
        all_labels = []

        for label, (attack_name, data_info) in enumerate(adversarial_data.items()):
            print(f"  Processing {attack_name} (label {label})...")
            patterns = self.extract_features(data_info['images'])
            all_patterns.append(patterns)
            all_labels.extend([label] * len(patterns))
            print(f"    ✓ Extracted {len(patterns)} patterns")

        X = np.vstack(all_patterns)
        y = np.array(all_labels)

        print(f"\n[INFO] Total training samples: {len(X)}")
        print(f"[INFO] Feature shape: {X.shape}")
        print(f"[INFO] Number of classes: {len(adversarial_data)}")

        return X, y

    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """訓練檢測器"""
        print(f"\n{'='*70}")
        print("Training Adversarial Detector")
        print(f"{'='*70}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")

        # 準備 DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_acc = 0
        best_epoch = 0

        for epoch in range(epochs):
            self.detector.train()
            train_loss = 0
            correct = 0
            total = 0

            for patterns, labels in train_loader:
                patterns = patterns.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.detector(patterns)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = 100. * correct / total
            avg_loss = train_loss / len(train_loader)

            # 測試
            test_acc = self.evaluate(X_test, y_test, batch_size)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Train Acc={train_acc:.2f}%, "
                      f"Test Acc={test_acc:.2f}%")

        print(f"\n{'='*70}")
        print(f"Training Completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        print(f"{'='*70}\n")

        return best_acc

    def evaluate(self, X, y, batch_size=100):
        """評估檢測器"""
        self.detector.eval()

        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for patterns, labels in loader:
                patterns = patterns.to(self.device)
                labels = labels.to(self.device)

                outputs = self.detector(patterns)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    def predict(self, X, batch_size=100):
        """預測樣本類別"""
        self.detector.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []
        probabilities = []

        num_batches = (len(X_tensor) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_tensor))
                batch = X_tensor[start_idx:end_idx]

                outputs = self.detector(batch)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return np.array(predictions), np.array(probabilities)

    def get_confusion_matrix(self, X, y_true, batch_size=100):
        """獲取混淆矩陣"""
        from sklearn.metrics import confusion_matrix

        y_pred, _ = self.predict(X, batch_size)
        cm = confusion_matrix(y_true, y_pred)

        return cm

    def get_per_class_accuracy(self, X, y_true, attack_types, batch_size=100):
        """獲取每類的準確率"""
        y_pred, _ = self.predict(X, batch_size)

        accuracies = {}
        for i, attack_name in enumerate(attack_types):
            mask = (y_true == i)
            if mask.sum() > 0:
                correct = (y_pred[mask] == y_true[mask]).sum()
                acc = 100. * correct / mask.sum()
                accuracies[attack_name] = acc
            else:
                accuracies[attack_name] = 0.0

        return accuracies

    def analyze_misclassifications(self, X, y_true, attack_types, batch_size=100):
        """分析誤分類情況"""
        y_pred, probs = self.predict(X, batch_size)

        print(f"\n{'='*70}")
        print("Misclassification Analysis")
        print(f"{'='*70}\n")

        for i, true_attack in enumerate(attack_types):
            true_mask = (y_true == i)
            if true_mask.sum() == 0:
                continue

            true_samples = y_pred[true_mask]
            misclassified = (true_samples != i)

            if misclassified.sum() > 0:
                print(f"\n{true_attack}:")
                print(f"  Total samples: {true_mask.sum()}")
                print(f"  Misclassified: {misclassified.sum()} ({100.*misclassified.sum()/true_mask.sum():.1f}%)")

                # 統計被誤分類為哪些類別
                misclassified_as = true_samples[misclassified]
                for j, pred_attack in enumerate(attack_types):
                    if j == i:
                        continue
                    count = (misclassified_as == j).sum()
                    if count > 0:
                        print(f"    → Misclassified as {pred_attack}: {count} ({100.*count/misclassified.sum():.1f}%)")

    def save_detector(self, path):
        """保存檢測器"""
        torch.save(self.detector.state_dict(), path)
        print(f"[INFO] Detector saved to {path}")

    def load_detector(self, path):
        """加載檢測器"""
        self.detector.load_state_dict(torch.load(path, map_location=self.device))
        self.detector.to(self.device)
        print(f"[INFO] Detector loaded from {path}")
