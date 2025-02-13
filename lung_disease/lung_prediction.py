import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time 
 

# màu nền
st.markdown(
    """
    <style>
    .stApp {
        background-color: #001a33;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#title
st.title(' 🫁 Dự đoán bệnh phổi từ ảnh ')
# Load model
# model = torch.load('model.pth')
# model.eval()

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize về kích thước phù hợp
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5])  
    ])
    return transform(image_path).unsqueeze(0)  # Thêm chiều cho ảnh

# Load image
uploaded_img = st.file_uploader(' 📤Chọn ảnh từ máy tính của bạn', type=['jpg', 'png', 'jpeg'])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert('RGB')

    # Bo góc ảnh & căn giữa
    st.markdown("""
        <style>
        .uploaded-img {
            display: flex;
            justify-content: center;
        }
        img {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hiển thị ảnh với chú thích đẹp
    st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
    st.image(image, caption="📷 Ảnh đã tải lên", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Preprocess image
    tensor = preprocess_image(image)

    if st.button('Dự đoán',  type="primary", icon="🔍"):
        st.markdown(" **Tiến hành phân tích ảnh...**")

#     # Hiệu ứng loading
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.01)  # Giả lập thời gian xử lý
#         progress.progress(i + 1)

#     # Dự đoán kết quả
#     input_tensor = preprocess_image(image)
#     with torch.no_grad():
#         output = model(tensor)
#         prediction = torch.argmax(output, 1).item()


#         classes = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']
#         st.success(f' # Dự đoán: {classes[prediction]}')
