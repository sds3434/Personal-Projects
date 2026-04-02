# GAD-7 · PHQ-9 · PCL-5 tracker (Streamlit)

Self-monitoring UI for symptom scales, medication log, daily habits, and charts. **Not medical advice or a diagnostic tool.**

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data files (local only)

The app creates CSV files in this folder for visits and logs. They are **gitignored** and are **not** pushed to GitHub:

- `sessions.csv` — scored surveys  
- `medications.csv` — medication events  
- `habits.csv` — daily habit log  

Clone the repo and run the app; new empty files are created as you use it.
