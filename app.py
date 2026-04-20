import gradio as gr
from ultralytics import YOLO
import os

# 1. 模型載入
model_path = "best.pt" 
if not os.path.exists(model_path):
    print("❌ 錯誤：找不到 best.pt")
else:
    model = YOLO(model_path)
    print(f"✅ 成功載入模型：{model_path}")

# 2. 辨識邏輯
def predict_image(img):
    if img is None:
        return None, "請先拍攝或上傳照片"
    
    results = model(img)
    probs = results[0].probs
    class_names = results[0].names
    
    # 取得最高機率
    top1_idx = probs.top1
    label_en = class_names[top1_idx]
    conf = probs.top1conf.item()
    
    # 翻譯成中文
    label_cn = "⚠️ 偵測到損壞 (DAMAGED)" if label_en == "damaged" else "✅ 貨物完好 (INTACT)"
    
    # 準備比例圖數據
    output_dict = {class_names[i]: float(probs.data[i]) for i in range(len(class_names))}
    
    return output_dict, f"判定結果：{label_cn}\n信心度：{conf*100:.2f}%"

# 3. 建立網頁介面 (企業級美化版)
with gr.Blocks(title="物流損壞辨識系統") as demo:
    # 使用 HTML 讓標題置中，看起來更專業
    gr.HTML("<h1 style='text-align: center; color: #2C3E50;'>📦 物流貨物損壞辨識系統</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 16px;'>請上傳貨物照片，AI 將自動分析紙箱是否發生破損。</p>")
    
    # ✨ 關鍵排版：使用 Row 讓大螢幕左右排列，手機自動上下折疊
    with gr.Row(): 
        
        # 左半邊：輸入區 (variant="panel" 會加上漂亮的灰色陰影外框)
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📸 步驟一：拍攝或上傳照片")
            input_img = gr.Image(
                type="pil", 
                label="圖片來源", 
                sources=["webcam", "upload"]
            )
            btn = gr.Button("🔍 開始 AI 分析", variant="primary", size="lg")
        
        # 右半邊：輸出區
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📊 步驟二：分析結果")
            output_label = gr.Label(num_top_classes=2, label="AI 信心度分析")
            # 加大文字框，讓結果更顯眼
            output_text = gr.Textbox(label="系統診斷結論", interactive=False, lines=2)

    # 觸發按鈕
    btn.click(fn=predict_image, inputs=input_img, outputs=[output_label, output_text])

# 4. 啟動分享模式
if __name__ == "__main__":
    # theme=gr.themes.Soft() 加上柔和的色彩主題
    demo.launch(share=True, theme=gr.themes.Soft())