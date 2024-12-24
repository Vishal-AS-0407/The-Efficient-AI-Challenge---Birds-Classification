# 🐦 Bird Classification Using EfficientNet-B7 and Distillation to EdgeNeXt

A highly accurate bird species classification system, leveraging state-of-the-art models like **EfficientNet-B7** and **EdgeNeXt xx-small**. This project is developed for the [Kaggle Bird Classification Competition](https://www.kaggle.com/competitions/bird-classification-anokha/data) and currently holds **Leaderboard #1**. 🚀

---


## 🗂️ Project Structure
```
.
├── effb7.py            # Teacher model training with EfficientNet-B7
├── distill.py          # Distillation into EdgeNeXt xx-small student model
├── test.py             # Final testing and submission file creation
├── FINALSUBMISSION     # Final submission file for Kaggle
└── annotation2.csv     # Dataset annotations
```

---

## 📚 Dataset Information
We used the **CUB-200-2011 Birds Dataset**, containing:
- **200 bird species**.
- **11,788 images**.

Download the dataset [here](http://www.vision.caltech.edu/datasets/cub_200_2011/).  
The dataset is split into training and testing sets using an 80-20 split.

---

## 🚀 Model Overview
### 1. **EfficientNet-B7 Teacher Model**
   - Pre-trained EfficientNet-B7 model fine-tuned on the dataset.
   - Achieved **state-of-the-art accuracy** during training.

### 2. **EdgeNeXt xx-small Student Model**
   - Distilled from the teacher model for lightweight inference.
   - Incorporates knowledge distillation with **temperature scaling**.

---

## 🛠️ Setup and Installation
### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- GPU (Recommended for training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/bird-classification.git
   cd bird-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the dataset:
   - Place images in the `images/` folder.
   - Ensure `annotation2.csv` is in the root directory.

---

## ⚡ How to Run
### Train the Teacher Model
```bash
python effb7.py
```

### Distill into the Student Model
```bash
python distill.py
```

### Generate Submission File
```bash
python test.py
```

---

## 🏅 Results and Achievements
- **Leaderboard #1** in the Kaggle competition! 🏆
---

## 📊 Performance Metrics

| Model                | Parameters     | Inference Time |
|----------------------|----------------|----------------|
| EfficientNet-B7      | 66M            | High           |
| EdgeNeXt xx-small    | 1.3M           | Low            |

---

### 🌟 **Show Your Support**
If you find this project helpful, give it a ⭐ on GitHub and share it with others! 😊
