

```markdown
# 🌿 Plant Disease Prediction using CNN

A Python-based image classifier built with a Convolutional Neural Network (CNN) to identify plant diseases from leaf images, leveraging the popular PlantVillage dataset.

---

## 🔍 Overview

This project provides a deep learning solution for detecting plant diseases using image classification. It uses CNNs to learn from a large dataset of healthy and diseased plant leaves, making it easier for farmers and researchers to identify and prevent crop damage early.

---

plant disease prediction/
├── .venv/                               # Python virtual environment
├── static/                              # Folder for static assets (CSS, images, JS)
│   └── ...                              # Add CSS/images as needed
├── templates/                           # Folder for HTML templates
│   ├── index.html                       # Main page with upload form
│   └── result.html                      # (Optional) Results display page
├── app.py                               # Main Flask app script
├── plant-disease-detection-using-cnn... # Jupyter notebook for model training


---

## 📦 Installation

### Requirements

- Python 3.7 or above
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
````

---

## 🧠 Dataset

* **Name**: PlantVillage Dataset
* **Source**: [Kaggle - PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage)
* **Classes**: \~38 plant disease and healthy categories
* **Preprocessing Steps**:

  * Image resizing (e.g. 224x224)
  * Normalization
  * Train-validation-test split

---

## 🧠 Model Architecture

* Convolutional Neural Network (CNN)
* Layers: Conv2D, MaxPooling2D, Dense, Dropout
* Optimizer: Adam
* Loss: Categorical Crossentropy
* Activation: ReLU + Softmax
* Evaluation: Accuracy, Loss plots, Confusion Matrix

---

## 🚀 Model Training

Open and run the Jupyter notebook:

```bash
jupyter notebook model_training_notebook/Plant_Disease_Training.ipynb
```

After training, the model is saved as `.h5` and used in the prediction app.

---

## 🧪 Running the Web App

Make sure the trained model file `plant_disease_model.h5` is inside `app/trained_model/`

Then run:

```bash
streamlit run app/main.py
```

or if not using Streamlit UI:

```bash
python app/main.py
```

Upload a leaf image to get disease predictions.

---

## 📊 Results

* Validation Accuracy: \~90–95%
* Output: Predicted class name (e.g., "Apple\_\_\_Black\_rot")
* Visual feedback with charts and predictions

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Streamlit (for web interface)
* Matplotlib, Seaborn

---

## 🧠 Future Enhancements

* Add transfer learning models like MobileNet, EfficientNet
* Deploy on cloud (Heroku, AWS, etc.)
* Real-time webcam support
* Mobile app integration

---

## 🤝 Contributing

Pull requests are welcome! You can contribute by:

* Improving model accuracy
* Enhancing the UI
* Cleaning and optimizing code

Fork the repo, create a branch, and submit a PR.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Muhammad Haseeb**
GitHub: [@Haseebf0164](https://github.com/Haseebf0164)

---

## 📚 References

* [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage)
* [CNN Image Classification](https://arxiv.org/abs/1604.03169)
* Similar GitHub projects using CNN for plant disease detection

---

**🌱 Empowering agriculture with deep learning.**


