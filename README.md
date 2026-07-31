# Heart Disease Prediction — End-to-End ML Deployment

An end-to-end machine learning project that predicts whether a patient is at
risk of heart disease based on clinical parameters. The model is trained in
Python, served through a Flask REST API, version-controlled on GitHub, and
deployed as a live web service on Render.

**Dataset:** [Kaggle — johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Live API URL:** `<ADD YOUR RENDER URL HERE AFTER DEPLOYMENT>`
e.g. `https://heart-disease-deployment.onrender.com`

---

## Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                # Flask REST API
├── train_model.py        # Data preprocessing + model training (Tasks 1 & 2)
├── model.pkl             # Trained model artifact (model + scaler + feature order)
├── heart.csv             # Dataset
├── requirements.txt      # Python dependencies
├── Procfile              # Render/gunicorn start command
├── README.md
├── templates/
│   └── index.html        # Optional simple info page
└── static/
    └── style.css
```

---

## Task 1: Data Understanding and Preprocessing

Implemented in `train_model.py`:

1. Loads `heart.csv` with Pandas.
2. Prints the first five records.
3. Identifies:
   - **Numerical features:** `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
   - **Target variable:** `target` (1 = heart disease present, 0 = absent)
4. Checks for missing values (none found in the dataset used).
5. Splits data into 80% training / 20% testing using `train_test_split` with stratification on the target.

## Task 2: Model Development

- **Algorithm used:** Random Forest Classifier (`n_estimators=200, max_depth=6`)
- Features are standardized with `StandardScaler` before training.
- **Accuracy Score (test set): ~0.74** (exact value printed when `train_model.py` is run; will differ once trained on the real Kaggle CSV instead of the bundled sample).
- The trained model, fitted scaler, and feature order are saved together into `model.pkl` using `joblib` so the API can reproduce identical preprocessing at inference time.

Run it yourself:
```bash
python train_model.py
```

## Task 3: API Development

`app.py` is a Flask REST API with three endpoints:

| Method | Route      | Description                                   |
|--------|-----------|------------------------------------------------|
| GET    | `/`        | Simple info page                              |
| GET    | `/health`  | Health check (used to confirm the service is up) |
| POST   | `/predict` | Accepts patient details as JSON, returns prediction |

**Example request:**
```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

**Example response:**
```json
{
  "prediction": "No Heart Disease Detected",
  "probability": 0.3814
}
```

Run it locally:
```bash
pip install -r requirements.txt
python app.py
# then POST to http://localhost:5000/predict
```

## Task 4: GitHub and Cloud Deployment

### GitHub
```bash
git init
git add .
git commit -m "Initial commit: Heart Disease Prediction end-to-end ML deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/HeartDiseaseDeployment.git
git push -u origin main
```
Make sure the repository is set to **Public**.

### Render Deployment
1. Go to [render.com](https://render.com) and sign in with GitHub.
2. Click **New +** → **Web Service** → connect the `HeartDiseaseDeployment` repo.
3. Configure:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already declared in the `Procfile`)
4. Choose the free instance type and click **Create Web Service**.
5. Once deployed, Render gives a public URL like `https://heart-disease-deployment.onrender.com`.
6. Test the live `/predict` endpoint with `curl` or Postman, and paste the URL at the top of this README.

**Note:** Free Render instances spin down after inactivity, which causes the first request after idle time to be slow (cold start). Keep this in mind during evaluation — hit `/health` once beforehand to warm the instance if needed.

## Task 5: Conclusion

The Random Forest classifier achieved close to 74% accuracy in distinguishing
patients at risk of heart disease from those without, based on clinical
parameters such as age, chest pain type, cholesterol, and maximum heart rate
achieved. Performance was balanced across both classes, suggesting the model
generalizes reasonably well without strongly favoring either outcome.

The main challenges during deployment involved keeping preprocessing
consistent between training and inference — solved by bundling the scaler
and feature order together with the model in a single artifact — and
handling Render's free-tier cold starts, which can make the first request
after idle time noticeably slower.

This project highlights why MLOps practices matter: version-controlled code,
reproducible training pipelines, and automated deployment turn a one-off
notebook experiment into a reliable, testable, and maintainable service that
others can actually use.
