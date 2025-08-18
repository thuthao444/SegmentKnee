import os
import sys
import time
import torch
import cv2
from torchvision import transforms
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MedViT import MedViT_tiny, MedViT_large  # hoặc MedViT_tiny nếu bạn dùng tiny
from ultrasound_dataset import UltrasoundTransform

# === CONFIG ===
checkpoint_path = './MedViT_large_knee2.pth'
model_type = 'MedViT_large'
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load mô hình 1 lần duy nhất ===
print("🔄 Loading model...")
model = MedViT_large(num_classes=num_classes).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
print("✅ Model loaded.")

# === Preprocessing transform ===
transform = UltrasoundTransform(is_training=False)
classes = ['Viêm', 'Không viêm']

# === Vòng lặp nhận ảnh liên tục ===
print("📂 Nhập đường dẫn ảnh PNG (hoặc gõ 'exit' để thoát):")

import cv2
import os

def visualize(image_path, pred_class, confidence):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    
    output_dir = "./results-normal"
    os.makedirs(output_dir, exist_ok=True)

    # Map class -> label
    class_map = {0: "abnormal", 1: "normal"}
    label = class_map.get(pred_class, str(pred_class))

    # Text hiển thị
    text = f"{label} | Conf: {confidence:.2f}"

    # Màu: đỏ cho Viêm (0), xanh cho Không viêm (1)
    color = (0, 0, 255) if pred_class == 0 else (0, 255, 0)

    # Vẽ chữ lên ảnh
    cv2.putText(
        img,
        text,
        (30, 50),                      # vị trí chữ
        cv2.FONT_HERSHEY_SIMPLEX,      # font
        1,                             # scale
        color,                         # màu
        2,                             # độ dày
        cv2.LINE_AA
    )

    # Lấy basename của ảnh gốc
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    save_path = os.path.join(output_dir, f"{name}_result{ext}")

    # Lưu ảnh
    cv2.imwrite(save_path, img)
    print(f"Saved result at {save_path}")
    return img

while True:
    try:
        image_path = input("🖼️ Path ảnh: ").strip()
        if image_path.lower() == "exit":
            print("👋 Kết thúc.")
            break

        if not os.path.isfile(image_path):
            print("❌ File không tồn tại. Thử lại.")
            continue

        # Load và transform ảnh
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Đo thời gian inference
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            visualize(image_path, pred_class, confidence)
        end_time = time.perf_counter()

        infer_time = (end_time - start_time) * 1000  # chuyển sang ms

        # Kết quả
        print(f"🔍 Prediction: {classes[pred_class]} (confidence: {confidence:.4f})")
        print(f"⏱️ Inference time: {infer_time:.2f} ms\n")

    except KeyboardInterrupt:
        print("\n👋 Đã thoát chương trình.")
        break
    except Exception as e:
        print(f"⚠️ Lỗi: {e}")
