import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = 'face_verifier_model.pth'

# ====== FaceNet и FaceVerifier ======
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_size)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

class FaceVerifier:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = FaceNet().to(device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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

@st.cache_resource
def load_verifier():
    return FaceVerifier(model_path=MODEL_PATH)

def load_image(label):
    uploaded_file = st.file_uploader(label=label, type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=label, use_column_width=True)
        # Сохраняем во временный файл для совместимости с FaceVerifier
        temp_path = f'temp_{label.replace(" ", "_")}.png'
        image.save(temp_path)
        return temp_path
    return None

st.title('Верификация лиц (Face Verification)')
st.write('Загрузите две фотографии для сравнения, чтобы определить, один ли это человек.')

img1_path = load_image('Фото 1')
img2_path = load_image('Фото 2')

if img1_path and img2_path:
    if st.button('Сравнить лица'):
        verifier = load_verifier()
        is_same, similarity = verifier.verify(img1_path, img2_path)
        st.write(f'**Схожесть:** {similarity:.4f}')
        st.write(f'**Вердикт:** {"Один человек" if is_same else "Разные люди"}')
        # Удаляем временные файлы
        os.remove(img1_path)
        os.remove(img2_path) 