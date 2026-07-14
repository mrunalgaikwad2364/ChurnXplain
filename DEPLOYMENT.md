# Deployment Guide

## 1. Deploy the Streamlit app (free, ~15 minutes)

### Step 1 — Push the project to GitHub
```bash
cd Churn_Proj
git init
git add .
git commit -m "Initial commit: churn prediction system"
```
Create a new repo on GitHub, then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

**Important:** `xgb_model.pkl` and `feature_schema.json` must be committed to the repo —
Streamlit Cloud only sees what's in GitHub. Don't gitignore them. (If your
model file is unusually large — XGBoost models here are typically only a
few MB, so this is fine — GitHub's normal 100MB file limit is the only
concern, and you're nowhere near it.)

Also commit `Dataset/Churn_Modelling.csv` if you want anyone (including
future-you) to be able to rerun `churn_training.py` from the repo, or add
it to `.gitignore` if you'd rather keep the raw dataset out of the repo
and just ship the trained model.

### Step 2 — Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repo, branch (`main`), and set the main file path to `churn_app.py`.
4. Click **Deploy**.

That's it — you'll get a public URL like `https://<your-app>.streamlit.app`.
Put this link directly on your resume next to the project title.

### Common deploy issues
- **"ModuleNotFoundError"** → check `requirements.txt` is in the repo root and lists every import used.
- **App crashes on load** → check the Streamlit Cloud logs (bottom-right "Manage app" panel); almost always a missing `xgb_model.pkl`/`feature_schema.json` in the repo, or a path issue.
- **Slow cold start** → normal on the free tier after inactivity; the first load after idle time takes 10-30s.

### Alternative: Hugging Face Spaces
If you'd rather use Hugging Face Spaces instead of Streamlit Cloud (also free):
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choose the **Streamlit** SDK.
2. Push your code to the Space's git repo the same way as above (HF gives you a git remote URL).
3. It builds and deploys automatically from `requirements.txt` + `churn_app.py`.

---

## 2. Run the test suite

```bash
pip install -r requirements.txt
python churn_training.py        # generates xgb_model.pkl + feature_schema.json, needed by most tests
pytest tests/ -v
```

Tests in `test_feature_engineering.py` run with no setup. Tests in
`test_model_utils.py` and `test_api.py` are automatically skipped if the
model/schema haven't been generated yet, so `pytest` never hard-fails in
a fresh clone before training.

### Optional: CI with GitHub Actions
Create `.github/workflows/tests.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```
This runs your test suite automatically on every push — a green checkmark
badge on the repo is a strong, low-effort signal for anyone reviewing it.

---

## 3. Run the FastAPI service locally

```bash
uvicorn api:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive API documentation
(FastAPI generates this automatically), or test it directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":650,"Geography":"France","Gender":"Female","Age":35,
       "Tenure":5,"Balance":50000,"NumOfProducts":2,"HasCrCard":1,
       "IsActiveMember":1,"EstimatedSalary":60000}'
```

The API and the Streamlit app share the exact same prediction logic
(`model_utils.py`), so they will always agree on a result for the same
customer — worth mentioning if this comes up in an interview.

### Deploying the API (optional, if you want a live API link too)
Free options: **Render** or **Railway**, both support deploying a FastAPI
app directly from a GitHub repo with a `Procfile` or auto-detected start
command (`uvicorn api:app --host 0.0.0.0 --port $PORT`).