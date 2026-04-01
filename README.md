# 🚀 Startup Profit Prediction System

A machine learning web application that predicts startup profit based on R&D Spend, Administration cost, Marketing Spend, and State — built with Flask and scikit-learn.

---

## 📁 Project Structure

```
startup_predictor/
├── web_app.py              # Flask backend & prediction API
├── requirements.txt        # Python dependencies
├── encoder                 # Pickled OneHotEncoder (State column)
├── scaler                  # Pickled StandardScaler (numeric columns)
├── model                   # Pickled regression model
└── templates/
    └── index.html          # Frontend UI
```

---

## ⚙️ Requirements

- Python 3.8+
- pip

---

## 🛠️ Installation

**1. Clone or download the project folder.**

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Make sure your model files are present in the root folder:**

```
encoder
scaler
model
```

---

## ▶️ Running the App

```bash
python web_app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

## 🔮 How It Works

1. The user enters **R&D Spend**, **Administration**, **Marketing Spend**, and selects a **State**.
2. The frontend sends the data as JSON to the `/predict` API endpoint via a POST request.
3. Flask preprocesses the input:
   - One-hot encodes the `State` column using the saved `encoder`
   - Scales numeric features using the saved `scaler`
4. The trained regression `model` predicts the profit.
5. The result is returned as JSON and displayed on the page.

---

## 🌐 API Reference

### `POST /predict`

**Request body (JSON):**

```json
{
  "rd_spend": 165349.20,
  "administration": 136897.80,
  "marketing_spend": 471784.10,
  "state": "California"
}
```

**Success response:**

```json
{
  "success": true,
  "profit": 192261.83
}
```

**Error response:**

```json
{
  "success": false,
  "error": "Error message here"
}
```

**Supported states:** `California`, `Florida`, `New York`

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError` on startup | Make sure `encoder`, `scaler`, and `model` files are in the same folder as `web_app.py` |
| `Network Error` in browser | Check that Flask is running and visit `http://localhost:5000` |
| Port already in use | Change the port: `app.run(debug=True, port=5001)` |
| Prediction fails | Ensure input values are valid numbers and state is one of the three supported options |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | scikit-learn |
| Data Processing | pandas |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Model Serialization | pickle |

---

## 📌 Notes

- The app runs in **debug mode** by default. Disable this for production by setting `debug=False`.
- The model was trained on the [50 Startups dataset](https://www.kaggle.com/datasets/farhanmd29/50-startups).
- Only three US states are supported: **California**, **Florida**, and **New York** — these are the states present in the training data.
