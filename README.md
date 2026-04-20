# 📦 物流貨物損壞辨識系統 (Logistics Damage Detection)

這是一個基於 **YOLOv8** 與 **Gradio** 開發的電腦視覺 Web 應用。
旨在協助物流人員透過手機或電腦鏡頭，即時辨識紙箱等貨物是否發生破損。

## ✨ 核心技術
*   **演算法**：YOLOv8 (影像分類模型)
*   **網頁框架**：Gradio
*   **準確度**：> 97% (使用自建物流資料集進行遷移學習)

## 🚀 如何在地端執行
1. 安裝環境依賴：
   ```bash
   pip install -r requirements.txt