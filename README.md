# 🤟 ASL Alphabet Images Detector Using DNN

A deep learning project that uses a **Dense Neural Network (DNN)** to classify images of the **American Sign Language (ASL) alphabet**.
The model is designed for high validation accuracy and low validation loss, using only **Dense layers** and the **Adam optimizer**.

---

## 📸 Preview

<p align="center">
<img width="1000" height="500" alt="ASL Alphabet Predictions" src="https://github.com/user-attachments/assets/1c1c2723-93e5-43a5-97ed-501b98adcff6" />
</p>

---

## ✨ Features

* 🔤 **ASL Alphabet Recognition** – Detects and classifies letters A–Z and special signs.
* 🧠 **DNN Architecture** – Built entirely with Dense layers and ReLU activations.
* ⚡ **Efficient Training** – Optimized with the Adam optimizer.
* 📊 **Performance Metrics** – Accuracy/loss curves.

---

## 🧠 Model Architecture

* **Input Layer**: Flattened image vectors
* **Hidden Layers**: Multiple dense layers with ReLU activation + Dropout
* **Output Layer**: 29 classes (A–Z + special signs)

---

## 📊 Results

* ✅ High validation accuracy
* 📉 Low validation loss

---

## ⚙️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/zeyadmedhat/ASL-Alphabet-Images-Detector-Using-DNN.git
   cd ASL-Alphabet-Images-Detector-Using-DNN
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data) and place it in the `dataset/` folder.

4. Train the model:

   ```bash
   python src/train.py
   ```

5. Evaluate the model:

   ```bash
   python src/evaluate.py
   ```

---

## 📌 Future Improvements

* 📷 Add **real-time hand gesture recognition** with webcam
* 🖥️ Deploy as a **web app or mobile app**
* 🔬 Explore **CNN architectures** for higher accuracy

---

## 🤝 Contributing

Contributions are welcome! Fork the repo and submit a pull request to suggest improvements.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Zeyad Medhat**
🔗 [GitHub Profile](https://github.com/zeyadmedhat)
