import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
from model import CNN
import os
import zipfile

# Streamlitウェブアプリのタイトル
st.title('CNN TruthFinder')
st.write('GANで生成された画像と本物の画像をCNNによって識別します。')

# PNGファイルのパスをリストとして定義
file_paths = ['gan_image1.png', 'gan_image2.png', 'true_image1.png', 'true_image2.png']

# ZIPファイルの名前を定義
zip_name = "test_images.zip"

# ZIPファイルを作成
with zipfile.ZipFile(zip_name, 'w') as zipf:
    for file_path in file_paths:
        # 各ファイルをZIPに追加
        zipf.write(file_path, os.path.basename(file_path))

# ZIPファイルをダウンロードボタンでダウンロードできるようにする
with open(zip_name, "rb") as file:
    st.download_button(
        label="テストデータをダウンロード",
        data=file,
        file_name=zip_name,
        mime='application/zip'
    )

# ユーザーがアップロードする画像の受け取り
uploaded_file = st.file_uploader("画像を選択してください...", type=["png", "jpg", "jpeg"])

# モデルの定義（前に定義したモデルクラス）
model = CNN()  # ここでCNNは事前に定義したモデルクラス
model.load_state_dict(torch.load('model.pth'))  # 学習済みの重みをロード
model.eval()

# 画像前処理の関数
def transform_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # グレースケール化
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# 画像がアップロードされたら
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # カラムの設定（左カラム、空白スペース、右カラム）
    col1, col_space, col2 = st.columns([5, 1, 5])  # 数値は相対的な幅
    
    # 左側のカラムに画像を表示
    with col1:
        st.image(image, caption='アップロードされた画像', use_column_width=True)
    
    # 右側のカラムに識別結果を表示
    with col2:
        st.write("識別結果...")
        
        # 画像をモデルに適用する準備
        image = transform_image(image)
        
        # 推論
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
        
        # 結果のテキストを表示
        if predicted.item() == 0:
            st.write('生成系AIで生成された画像です。')
        else:
            st.write('本物の画像です。')
        
        # # 推論されたクラスを表示
        # st.write(f'予測されたクラス: {predicted.item()}')
