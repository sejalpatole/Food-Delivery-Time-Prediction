# 🍕 AI-Powered Food Delivery Time Prediction

<p align="center">
  <b>Predict food delivery time using Machine Learning and Streamlit.</b><br>
  Built with Python, Scikit-learn, and an interactive web interface.
</p>

---

## 📖 Overview

This project predicts the estimated delivery time for food orders based on delivery partner information, weather conditions, traffic density, vehicle type, city, and several other real-world factors.

The application uses a trained Machine Learning model to generate delivery time predictions instantly through an intuitive Streamlit interface.

---

## 📸 Project Preview

### 🏠 Home Dashboard

![Dashboard](screenshots/dashboard.png)

### 🚚 Prediction Result

![Prediction](screenshots/prediction.png)

---

## ✨ Features

* 🤖 Machine Learning-based delivery time prediction
* 📍 Automatic distance calculation using the Haversine Formula
* 🌦 Weather and traffic condition analysis
* 🚗 Vehicle and order type selection
* 🏙 City-based prediction
* 📅 Order and pickup time inputs
* 📊 Instant prediction results
* 📈 Interactive Streamlit dashboard
* 💎 Clean, modern, responsive user interface

---

## 🛠️ Technologies Used

| Category         | Tools         |
| ---------------- | ------------- |
| Programming      | Python        |
| Machine Learning | Scikit-learn  |
| Data Analysis    | Pandas, NumPy |
| Visualization    | Plotly        |
| Web App          | Streamlit     |

---

## 📂 Project Structure

```text
Food-Delivery-Time-Prediction/
│
├── app.py
├── food_delivery_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── train.csv
├── test.csv
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── dashboard.png
│   └── prediction.png
│
└── Food Delivery Time Prediction.ipynb
```

---

## 📊 Model Features

The model uses the following inputs:

* Delivery Person Age
* Delivery Person Ratings
* Restaurant Latitude & Longitude
* Delivery Location Latitude & Longitude
* Weather Conditions
* Road Traffic Density
* Vehicle Condition
* Type of Order
* Type of Vehicle
* Multiple Deliveries
* Festival
* City
* Order Date
* Order Time
* Pickup Time
* Distance (Calculated)

---

## ⚙️ Machine Learning Pipeline

1. Data Cleaning
2. Missing Value Handling
3. Feature Engineering
4. Label Encoding
5. Distance Calculation (Haversine Formula)
6. Feature Scaling
7. Random Forest Model Training
8. Model Evaluation
9. Streamlit Deployment

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/sejalpatole/Food-Delivery-Time-Prediction.git
```

### Navigate to the project folder

```bash
cd Food-Delivery-Time-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📈 Future Enhancements

* Google Maps API integration
* Real-time Weather API
* Live Traffic API
* Interactive route visualization
* Prediction history
* Model optimization
* Cloud deployment improvements

---

## 👩‍💻 Author

**Sejal Patole**

Aspiring Machine Learning Engineer passionate about building practical AI-powered applications using Python and Machine Learning.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.
