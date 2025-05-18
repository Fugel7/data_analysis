# Функция для сравнения загруженных фотографий
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import random
import matplotlib.pyplot as plt

def compare_uploaded_photos(model_path='face_verifier_model.pth'):
    try:
        from google.colab import files
        import matplotlib.pyplot as plt
    except ImportError:
        print("Эта функция работает только в Google Colab")
        return

    print("Загрузите первое фото...")
    uploaded1 = files.upload()
    if not uploaded1:
        print("Ошибка: первое фото не загружено")
        return

    print("\nЗагрузите второе фото...")
    uploaded2 = files.upload()
    if not uploaded2:
        print("Ошибка: второе фото не загружено")
        return

    img1_path = list(uploaded1.keys())[0]
    img2_path = list(uploaded2.keys())[0]

    verifier = FaceVerifier()

    if os.path.exists(model_path):
        try:
            verifier.model.load_state_dict(torch.load(model_path))
            print(f"\nЗагружена обученная модель из {model_path}")
        except Exception as e:
            print(f"\nОшибка загрузки модели: {e}")
            print("Используется предобученная модель")
    else:
        print("\nИспользуется предобученная модель (без дообучения)")

    try:
        is_same, similarity = verifier.verify(img1_path, img2_path)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        img1 = Image.open(img1_path)
        plt.imshow(img1)
        plt.axis('off')
        plt.title("Фото 1", pad=10)

        plt.subplot(1, 2, 2)
        img2 = Image.open(img2_path)
        plt.imshow(img2)
        plt.axis('off')
        plt.title("Фото 2", pad=10)

        plt.tight_layout(pad=3.0)
        plt.show()

        print(f"\nРезультаты сравнения:")
        print(f"Схожесть: {similarity:.4f}")
        print(f"Вердикт модели: {'Один человек' if is_same else 'Разные люди'}")

    except Exception as e:
        print(f"Ошибка при обработке фотографий: {e}")

    # Удаляем временные файлы
    try:
        os.remove(img1_path)
        os.remove(img2_path)
    except:
        pass


# =============== Архитектура сети ===============
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()

        # Основные сверточные слои
        self.conv_layers = nn.Sequential(
            # 1: 160x160x3 -> 80x80x32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 2: 80x80x32 -> 40x40x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 3: 40x40x64 -> 20x20x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 4: 20x20x128 -> 10x10x256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 5: 10x10x256 -> 5x5x512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Полносвязные слои для эмбеддинга
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        # Проход через сверточные слои
        x = self.conv_layers(x)

        # Преобразование в вектор
        x = x.view(x.size(0), -1)

        # Получение эмбеддинга
        x = self.fc_layers(x)

        # L2 нормализация
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

class FaceVerifier:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = FaceNet().to(device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            face = self.transform(img).unsqueeze(0).to(self.device)
            return face
        except Exception as e:
            print(f"Ошибка обработки изображения {img_path}: {str(e)}")
            return None

    def get_embedding(self, img_path):
        face = self.preprocess_image(img_path)
        if face is None:
            return None

        with torch.no_grad():
            embedding = self.model(face)
        return embedding.cpu().numpy()

    def verify(self, img1_path, img2_path, threshold=0.7):
        emb1 = self.get_embedding(img1_path)
        emb2 = self.get_embedding(img2_path)

        if emb1 is None or emb2 is None:
            return False, 0.0

        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity > threshold, similarity

class FaceVerifierTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = FaceNet().to(device)
        self.criterion = nn.TripletMarginLoss(margin=0.3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            face = self.transform(img).to(self.device)
            return face
        except Exception as e:
            print(f"Ошибка обработки изображения {img_path}: {str(e)}")
            return None

    def train_epoch(self, triplets, batch_size=32):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Предобработка триплетов
        processed_triplets = []
        for anchor_path, positive_path, negative_path in tqdm(triplets, desc="Предобработка"):
            try:
                anchor_face = self.preprocess_image(anchor_path)
                positive_face = self.preprocess_image(positive_path)
                negative_face = self.preprocess_image(negative_path)

                if anchor_face is not None and positive_face is not None and negative_face is not None:
                    processed_triplets.append((anchor_face, positive_face, negative_face))
            except Exception as e:
                print(f"Ошибка обработки триплета: {str(e)}")
                continue

        if not processed_triplets:
            print("Нет валидных триплетов!")
            return 0, 0

        # Обучение по батчам
        num_batches = len(processed_triplets) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = processed_triplets[start_idx:end_idx]

            # Подготовка батча
            try:
                anchors = torch.stack([t[0] for t in batch])
                positives = torch.stack([t[1] for t in batch])
                negatives = torch.stack([t[2] for t in batch])

                # Прямой проход
                anchor_emb = self.model(anchors)
                positive_emb = self.model(positives)
                negative_emb = self.model(negatives)

                # Вычисление лосса и обратное распространение
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Подсчет точности
                with torch.no_grad():
                    pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
                    neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
                    correct += (pos_dist < neg_dist).sum().item()
                    total += len(anchors)
            except Exception as e:
                print(f"Ошибка при обработке батча {i}: {str(e)}")
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy
