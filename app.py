"""
MediCost AI - Health insurance cost estimator
---------------------------------------------
Flask app that serves an XGBoost regression model (xgboost.pkl) trained on the
medical-insurance dataset (age, sex, bmi, children, smoker, region -> charges).

Features
  * Live prediction while you move the sliders
  * "Why this price?" explanation using XGBoost SHAP contributions
  * What-if simulator (quit smoking, healthy BMI, ageing, ...)
  * Cost-vs-age curve for smokers and non-smokers
  * USD / INR toggle, BMI calculator from height & weight

Run locally:   python app.py
Run on Render: gunicorn app:app
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("medicost")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "xgboost.pkl"))

# Feature order must match the order the model was trained with.
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]

# Label-encoding used at training time (alphabetical, i.e. sklearn LabelEncoder).
# If your notebook used a different mapping, change it here.
SEX_MAP = {"female": 0, "male": 1}
SMOKER_MAP = {"no": 0, "yes": 1}
REGION_MAP = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

AGE_MIN, AGE_MAX = 18, 64  # range the model was trained on

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        log.info("Model loaded from %s (%s)", MODEL_PATH, type(model).__name__)
        return model
    except Exception as exc:  # keep the site alive so /health can report the problem
        log.exception("Could not load model: %s", exc)
        return None


model = load_model()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_frame(rows):
    df = pd.DataFrame(rows, columns=FEATURES)
    return df.astype(
        {
            "age": "int64",
            "sex": "int64",
            "bmi": "float64",
            "children": "int64",
            "smoker": "int64",
            "region": "int64",
        }
    )


def predict_rows(rows):
    return [float(v) for v in model.predict(make_frame(rows))]


def contributions(row):
    """Per-feature SHAP values (in $) for one row, plus the baseline."""
    try:
        import xgboost as xgb

        dm = xgb.DMatrix(make_frame([row]))
        contribs = model.get_booster().predict(dm, pred_contribs=True)[0]
        return [float(c) for c in contribs[:-1]], float(contribs[-1])
    except Exception as exc:
        log.warning("SHAP contributions unavailable: %s", exc)
        return None, None


def parse_payload(data):
    try:
        age = int(data.get("age"))
        bmi = round(float(data.get("bmi")), 1)
        children = int(data.get("children"))
        sex = str(data.get("sex")).lower()
        smoker = str(data.get("smoker")).lower()
        region = str(data.get("region")).lower()
    except (TypeError, ValueError):
        raise ValueError("Please fill in all fields with valid numbers.")

    if not AGE_MIN <= age <= AGE_MAX:
        raise ValueError(f"Age must be between {AGE_MIN} and {AGE_MAX}.")
    if not 15 <= bmi <= 55:
        raise ValueError("BMI must be between 15 and 55.")
    if not 0 <= children <= 5:
        raise ValueError("Children must be between 0 and 5.")
    if sex not in SEX_MAP or smoker not in SMOKER_MAP or region not in REGION_MAP:
        raise ValueError("Invalid sex, smoker or region value.")

    return {
        "age": age,
        "sex": SEX_MAP[sex],
        "bmi": bmi,
        "children": children,
        "smoker": SMOKER_MAP[smoker],
        "region": REGION_MAP[region],
    }


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def risk_tier(cost):
    if cost < 6000:
        return "Low"
    if cost < 15000:
        return "Moderate"
    if cost < 30000:
        return "High"
    return "Very high"


def feature_label(name, row):
    inv = lambda m, v: next(k for k, x in m.items() if x == v).title()
    return {
        "age": f"Age {row['age']}",
        "sex": inv(SEX_MAP, row["sex"]),
        "bmi": f"BMI {row['bmi']}",
        "children": f"{row['children']} child" + ("" if row["children"] == 1 else "ren"),
        "smoker": "Smoker" if row["smoker"] else "Non-smoker",
        "region": inv(REGION_MAP, row["region"]),
    }[name]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/health")
def health():
    return jsonify(status="ok" if model is not None else "model_not_loaded"), (
        200 if model is not None else 503
    )


@app.post("/api/predict")
def api_predict():
    if model is None:
        return jsonify(error="Model is not available on the server."), 503
    try:
        row = parse_payload(request.get_json(force=True, silent=True) or {})
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    # ---- build every scenario we need and predict them in one batch ----
    scenarios = [("base", row)]
    if row["smoker"]:
        scenarios.append(("quit", {**row, "smoker": 0}))
    else:
        scenarios.append(("start", {**row, "smoker": 1}))
    if row["bmi"] >= 25:
        scenarios.append(("bmi", {**row, "bmi": 23.0}))
    older_age = min(row["age"] + 10, AGE_MAX)
    if older_age > row["age"]:
        scenarios.append(("older", {**row, "age": older_age}))

    ages = list(range(AGE_MIN, AGE_MAX + 1))
    curve_rows = [{**row, "age": a, "smoker": 1} for a in ages] + [
        {**row, "age": a, "smoker": 0} for a in ages
    ]

    all_rows = [r for _, r in scenarios] + curve_rows
    preds = predict_rows(all_rows)
    scen_pred = dict(zip([k for k, _ in scenarios], preds[: len(scenarios)]))
    curve_pred = preds[len(scenarios):]

    estimate = max(scen_pred["base"], 0.0)

    cards = []
    if "quit" in scen_pred:
        cards.append(("Quit smoking", "Biggest single saving available", "quit", scen_pred["quit"]))
    if "start" in scen_pred:
        cards.append(("If you started smoking", "Why staying smoke-free matters", "start", scen_pred["start"]))
    if "bmi" in scen_pred:
        cards.append(("Reach a healthy BMI (23)", "Weight management effect", "bmi", scen_pred["bmi"]))
    if "older" in scen_pred:
        cards.append((f"In {older_age - row['age']} years (age {older_age})", "Costs rise as you get older", "older", scen_pred["older"]))

    # ---- explanation (SHAP) ----
    contribs, baseline = contributions(row)
    explain = None
    if contribs is not None:
        explain = [
            {"feature": f, "label": feature_label(f, row), "value": round(c, 2)}
            for f, c in zip(FEATURES, contribs)
        ]
        explain.sort(key=lambda d: abs(d["value"]), reverse=True)

    return jsonify(
        estimate=round(estimate, 2),
        monthly=round(estimate / 12, 2),
        baseline=round(baseline, 2) if baseline is not None else 13270.0,
        tier=risk_tier(estimate),
        bmi_category=bmi_category(row["bmi"]),
        explain=explain,
        scenarios=[
            {
                "key": key,
                "title": title,
                "desc": desc,
                "cost": round(max(cost, 0.0), 2),
                "delta": round(max(cost, 0.0) - estimate, 2),
            }
            for title, desc, key, cost in cards
        ],
        curve={
            "ages": ages,
            "smoker": [round(max(v, 0.0), 2) for v in curve_pred[: len(ages)]],
            "nonsmoker": [round(max(v, 0.0), 2) for v in curve_pred[len(ages):]],
        },
    )


# --------------------------------------------------------------------------- #
# Front-end (single page, no external JS dependencies)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MediCost AI · Health insurance cost estimator</title>
<meta name="description" content="Estimate your yearly health insurance cost with an XGBoost machine-learning model. See what drives the price and how to lower it.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🩺</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#070b17; --panel:rgba(255,255,255,.055); --panel-b:rgba(255,255,255,.11);
  --text:#eaf0ff; --muted:#93a0c0; --accent:#5eead4; --accent2:#818cf8;
  --good:#34d399; --bad:#fb7185; --warn:#fbbf24;
  --radius:20px;
  --font:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --display:'Space Grotesk','Inter',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0}
html{scroll-behavior:smooth}
body{
  font-family:var(--font);color:var(--text);background:var(--bg);min-height:100vh;
  background-image:
    radial-gradient(900px 500px at 8% -10%,rgba(94,234,212,.18),transparent 60%),
    radial-gradient(800px 520px at 100% 0%,rgba(129,140,248,.22),transparent 60%),
    radial-gradient(700px 500px at 50% 110%,rgba(244,114,182,.10),transparent 60%);
  background-attachment:fixed;
  padding:env(safe-area-inset-top,0) 0 40px;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
header{display:flex;align-items:center;justify-content:space-between;padding:26px 0 8px;gap:12px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;font-size:22px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 8px 28px rgba(94,234,212,.28)}
.brand b{font-family:var(--display);font-size:21px;letter-spacing:-.3px}
.brand span{display:block;font-size:12px;color:var(--muted);margin-top:1px}
.chip{font-size:12px;color:var(--muted);padding:7px 12px;border:1px solid var(--panel-b);border-radius:999px;background:var(--panel)}
.chip i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--good);margin-right:7px;box-shadow:0 0 10px var(--good)}

.hero{padding:34px 0 26px;max-width:760px}
.hero h1{font-family:var(--display);font-size:clamp(32px,5vw,54px);line-height:1.05;letter-spacing:-1.4px}
.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:var(--muted);margin-top:14px;font-size:17px;line-height:1.55}

.grid{display:grid;grid-template-columns:420px 1fr;gap:22px;align-items:start}
@media(max-width:960px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--panel-b);border-radius:var(--radius);padding:24px;
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);box-shadow:0 20px 60px rgba(0,0,0,.35)}
.card h2{font-family:var(--display);font-size:18px;letter-spacing:-.2px;margin-bottom:4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.stack{display:flex;flex-direction:column;gap:22px}

.field{margin-bottom:20px}
.field:last-child{margin-bottom:0}
.lbl{display:flex;justify-content:space-between;align-items:baseline;font-size:13px;color:var(--muted);margin-bottom:10px;font-weight:500}
.lbl output{color:var(--text);font-family:var(--display);font-size:20px;font-weight:700}
.lbl small{font-weight:600;margin-left:8px;font-size:12px}

input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:8px;border-radius:999px;outline:none;
  background:linear-gradient(90deg,var(--accent) var(--p,50%),rgba(255,255,255,.12) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;border-radius:50%;background:#fff;
  border:5px solid var(--accent);box-shadow:0 4px 14px rgba(0,0,0,.5);cursor:pointer;transition:transform .15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.12)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:5px solid var(--accent);cursor:pointer}

.seg{display:grid;gap:6px;background:rgba(0,0,0,.28);padding:5px;border-radius:14px}
.seg.c2{grid-template-columns:1fr 1fr}.seg.c4{grid-template-columns:1fr 1fr}
.seg button{border:0;background:transparent;color:var(--muted);font:600 14px var(--font);padding:11px 8px;border-radius:10px;cursor:pointer;transition:.18s}
.seg button:hover{color:var(--text)}
.seg button.on{background:linear-gradient(135deg,rgba(94,234,212,.25),rgba(129,140,248,.28));color:#fff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.18)}
.seg button.on.danger{background:linear-gradient(135deg,rgba(251,113,133,.3),rgba(251,191,36,.2))}

.stepper{display:flex;align-items:center;gap:14px}
.stepper button{width:40px;height:40px;border-radius:12px;border:1px solid var(--panel-b);background:rgba(255,255,255,.06);color:var(--text);font-size:20px;cursor:pointer;transition:.15s}
.stepper button:hover{background:rgba(255,255,255,.14)}
.stepper .kids{display:flex;gap:6px;font-size:22px;min-width:120px;flex-wrap:wrap}

details{margin-top:6px;border:1px dashed var(--panel-b);border-radius:14px;padding:12px 14px}
summary{cursor:pointer;font-size:13px;color:var(--muted);font-weight:500}
.bmi-calc{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
.bmi-calc label{font-size:12px;color:var(--muted)}
.bmi-calc input{width:100%;margin-top:5px;background:rgba(0,0,0,.3);border:1px solid var(--panel-b);border-radius:10px;padding:10px;color:var(--text);font:500 15px var(--font)}
.bmi-calc input:focus{outline:2px solid var(--accent);outline-offset:1px}

/* result */
.result{position:relative;overflow:hidden}
.result::before{content:"";position:absolute;inset:-40% -20% auto auto;width:360px;height:360px;border-radius:50%;
  background:radial-gradient(circle,rgba(94,234,212,.22),transparent 65%);pointer-events:none}
.res-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
.big{font-family:var(--display);font-size:clamp(46px,8vw,76px);font-weight:700;letter-spacing:-2.5px;line-height:1.05;margin:8px 0 2px;
  background:linear-gradient(180deg,#fff,#b8c6ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.per{color:var(--muted);font-size:15px}
.per b{color:var(--text)}
.cur{display:flex;background:rgba(0,0,0,.3);border-radius:999px;padding:4px}
.cur button{border:0;background:transparent;color:var(--muted);font:600 12px var(--font);padding:7px 14px;border-radius:999px;cursor:pointer}
.cur button.on{background:#fff;color:#0b1020}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.badge{font-size:12px;font-weight:600;padding:6px 12px;border-radius:999px;border:1px solid var(--panel-b);background:rgba(255,255,255,.06)}
.t-Low{color:var(--good);border-color:rgba(52,211,153,.4)}
.t-Moderate{color:var(--warn);border-color:rgba(251,191,36,.4)}
.t-High,.t-Very{color:var(--bad);border-color:rgba(251,113,133,.45)}

.meter{margin-top:26px}
.meter-bar{position:relative;height:12px;border-radius:999px;background:linear-gradient(90deg,#34d399,#fbbf24 45%,#fb7185)}
.meter-dot{position:absolute;top:50%;width:24px;height:24px;border-radius:50%;background:#fff;border:5px solid #0b1020;
  transform:translate(-50%,-50%);left:10%;transition:left .6s cubic-bezier(.2,.9,.3,1);box-shadow:0 0 0 2px #fff,0 6px 18px rgba(0,0,0,.5)}
.meter-avg{position:absolute;top:-6px;bottom:-6px;width:2px;background:rgba(255,255,255,.7)}
.meter-avg span{position:absolute;top:-22px;left:50%;transform:translateX(-50%);font-size:11px;color:var(--muted);white-space:nowrap}
.meter-foot{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:12px}
.vs{margin-top:10px;font-size:14px;color:var(--muted)}
.vs b.up{color:var(--bad)}.vs b.down{color:var(--good)}

/* explain */
.row{display:grid;grid-template-columns:130px 1fr 92px;gap:12px;align-items:center;padding:9px 0;font-size:14px}
.row+.row{border-top:1px solid rgba(255,255,255,.06)}
.row .name{color:var(--text);font-weight:500}
.row .val{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.track{position:relative;height:10px;background:rgba(255,255,255,.07);border-radius:999px}
.track::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:rgba(255,255,255,.25)}
.fill{position:absolute;top:0;bottom:0;border-radius:999px;transition:width .6s cubic-bezier(.2,.9,.3,1)}
.fill.pos{left:50%;background:linear-gradient(90deg,#fb7185,#f97316)}
.fill.neg{right:50%;background:linear-gradient(270deg,#34d399,#2dd4bf)}
.val.pos{color:#fda4af}.val.neg{color:#6ee7b7}
.base-row .name{color:var(--muted)}
.hint{margin-top:12px;font-size:12px;color:var(--muted);line-height:1.5}

/* scenarios */
.scen{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.sc{padding:18px;border-radius:16px;background:rgba(0,0,0,.25);border:1px solid var(--panel-b);transition:transform .2s,border-color .2s}
.sc:hover{transform:translateY(-3px);border-color:rgba(255,255,255,.3)}
.sc h3{font-size:14px;font-weight:600}
.sc p{font-size:12px;color:var(--muted);margin:3px 0 12px}
.sc .c{font-family:var(--display);font-size:25px;font-weight:700;letter-spacing:-.6px}
.pill{display:inline-block;margin-top:8px;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700}
.pill.save{background:rgba(52,211,153,.15);color:#6ee7b7}
.pill.more{background:rgba(251,113,133,.15);color:#fda4af}

/* chart */
svg{display:block;width:100%;height:auto;overflow:visible}
.legend{display:flex;gap:18px;font-size:12px;color:var(--muted);margin-top:10px;flex-wrap:wrap}
.legend i{display:inline-block;width:12px;height:4px;border-radius:2px;margin-right:7px;vertical-align:middle}

.err{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:#3b1220;border:1px solid var(--bad);color:#fecdd3;
  padding:12px 18px;border-radius:12px;font-size:14px;display:none;z-index:9;max-width:90vw}
.loading .big{opacity:.55}
footer{margin-top:34px;color:var(--muted);font-size:12px;line-height:1.6;text-align:center}
footer b{color:var(--text)}
@media(max-width:520px){.row{grid-template-columns:100px 1fr 80px}.card{padding:18px}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">🩺</div>
      <div><b>MediCost AI</b><span>Health insurance cost estimator</span></div>
    </div>
    <div class="chip"><i></i>XGBoost model · 100 trees · live</div>
  </header>

  <section class="hero">
    <h1>Know what your health cover will <em>really</em> cost.</h1>
    <p>Move the sliders and watch the machine-learning estimate update instantly. See what drives your price and what you could change to lower it.</p>
  </section>

  <div class="grid">
    <!-- INPUTS -->
    <div class="card" id="form">
      <h2>Your profile</h2>
      <p class="sub">Adjust anything: the estimate updates live.</p>

      <div class="field">
        <div class="lbl"><span>Age</span><output id="ageOut">35</output></div>
        <input type="range" id="age" min="18" max="64" value="35">
      </div>

      <div class="field">
        <div class="lbl"><span>Body Mass Index (BMI)</span><span><output id="bmiOut">27.0</output><small id="bmiCat"></small></span></div>
        <input type="range" id="bmi" min="15" max="50" step="0.1" value="27">
        <details>
          <summary>Don't know your BMI? Calculate it</summary>
          <div class="bmi-calc">
            <label>Height (cm)<input type="number" id="hcm" min="120" max="230" placeholder="170"></label>
            <label>Weight (kg)<input type="number" id="wkg" min="30" max="250" placeholder="70"></label>
          </div>
        </details>
      </div>

      <div class="field">
        <div class="lbl"><span>Biological sex</span></div>
        <div class="seg c2" data-group="sex">
          <button data-v="female" class="on">♀ Female</button>
          <button data-v="male">♂ Male</button>
        </div>
      </div>

      <div class="field">
        <div class="lbl"><span>Do you smoke?</span></div>
        <div class="seg c2" data-group="smoker">
          <button data-v="no" class="on">🌿 No</button>
          <button data-v="yes" class="danger">🚬 Yes</button>
        </div>
      </div>

      <div class="field">
        <div class="lbl"><span>Children covered</span></div>
        <div class="stepper">
          <button id="kidsMinus" aria-label="Fewer children">−</button>
          <div class="kids" id="kidsIcons"></div>
          <button id="kidsPlus" aria-label="More children">+</button>
        </div>
      </div>

      <div class="field">
        <div class="lbl"><span>Region</span></div>
        <div class="seg c4" data-group="region">
          <button data-v="northeast">Northeast</button>
          <button data-v="northwest">Northwest</button>
          <button data-v="southeast" class="on">Southeast</button>
          <button data-v="southwest">Southwest</button>
        </div>
      </div>
    </div>

    <!-- RESULTS -->
    <div class="stack">
      <div class="card result" id="result">
        <div class="res-top">
          <div>
            <div class="eyebrow">Estimated yearly cost</div>
            <div class="big" id="estimate">$0</div>
            <div class="per">≈ <b id="monthly">$0</b> per month</div>
          </div>
          <div class="cur" id="cur">
            <button data-c="USD" class="on">USD $</button>
            <button data-c="INR">INR ₹</button>
          </div>
        </div>
        <div class="badges">
          <span class="badge" id="tier">Risk tier</span>
          <span class="badge" id="bmiBadge">BMI</span>
        </div>
        <div class="meter">
          <div class="meter-bar">
            <div class="meter-avg" id="avgMark"><span>Average</span></div>
            <div class="meter-dot" id="dot"></div>
          </div>
          <div class="meter-foot"><span id="mMin"></span><span id="mMax"></span></div>
        </div>
        <div class="vs" id="vs"></div>
      </div>

      <div class="card" id="explainCard">
        <h2>Why this price?</h2>
        <p class="sub">How each factor pushes your estimate above or below the average policyholder (SHAP values from the model).</p>
        <div id="explain"></div>
        <p class="hint">Red bars increase your cost, green bars reduce it. Values add up to your final estimate.</p>
      </div>

      <div class="card">
        <h2>What if…?</h2>
        <p class="sub">Compare your estimate against realistic changes.</p>
        <div class="scen" id="scen"></div>
      </div>

      <div class="card">
        <h2>Cost across your lifetime</h2>
        <p class="sub">Same profile as yours, only age (and smoking status) changes.</p>
        <div id="chart"></div>
        <div class="legend">
          <span><i style="background:#fb7185"></i>Smoker</span>
          <span><i style="background:#5eead4"></i>Non-smoker</span>
          <span><i style="background:#fff"></i>You</span>
        </div>
      </div>
    </div>
  </div>

  <footer>
    <b>Educational estimate only.</b> Predictions come from a machine-learning model trained on a public medical-insurance dataset (US, USD).
    Real premiums depend on insurer, plan and medical history. INR uses an indicative fixed rate.<br>
    Built with Flask &amp; XGBoost.
  </footer>
</div>
<div class="err" id="err"></div>

<script>
const $ = s => document.querySelector(s);
const INR_RATE = 88;                 // indicative USD -> INR rate, edit as needed
const state = {sex:'female', smoker:'no', region:'southeast', kids:0, currency:'USD'};
let last = null, shown = 0, timer = null, ctrl = null;

function money(usd, dec = 0){
  const inr = state.currency === 'INR';
  return new Intl.NumberFormat(inr ? 'en-IN' : 'en-US',
    {style:'currency', currency:state.currency, maximumFractionDigits:dec}).format(inr ? usd * INR_RATE : usd);
}
function signed(usd){ return (usd >= 0 ? '+' : '−') + money(Math.abs(usd)); }

/* ---------- inputs ---------- */
function paintRange(el){ el.style.setProperty('--p', ((el.value - el.min) / (el.max - el.min) * 100) + '%'); }
function bmiInfo(b){
  if (b < 18.5) return ['Underweight', '#fbbf24'];
  if (b < 25)   return ['Healthy', '#34d399'];
  if (b < 30)   return ['Overweight', '#fbbf24'];
  return ['Obese', '#fb7185'];
}
function syncLabels(){
  $('#ageOut').textContent = $('#age').value;
  const b = parseFloat($('#bmi').value);
  $('#bmiOut').textContent = b.toFixed(1);
  const [t, c] = bmiInfo(b);
  const s = $('#bmiCat'); s.textContent = t; s.style.color = c;
  paintRange($('#age')); paintRange($('#bmi'));
  $('#kidsIcons').textContent = state.kids ? '🧒'.repeat(state.kids) : 'None';
}
function schedule(){ syncLabels(); clearTimeout(timer); timer = setTimeout(predict, 220); }

['age','bmi'].forEach(id => $('#'+id).addEventListener('input', schedule));
document.querySelectorAll('.seg').forEach(seg => {
  seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    seg.querySelectorAll('button').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); state[seg.dataset.group] = b.dataset.v; schedule();
  });
});
$('#kidsMinus').onclick = () => { state.kids = Math.max(0, state.kids - 1); schedule(); };
$('#kidsPlus').onclick  = () => { state.kids = Math.min(5, state.kids + 1); schedule(); };
function bmiFromBody(){
  const h = parseFloat($('#hcm').value), w = parseFloat($('#wkg').value);
  if (h > 100 && w > 20){
    const b = Math.min(50, Math.max(15, w / ((h/100) ** 2)));
    $('#bmi').value = b.toFixed(1); schedule();
  }
}
$('#hcm').addEventListener('input', bmiFromBody);
$('#wkg').addEventListener('input', bmiFromBody);
$('#cur').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  document.querySelectorAll('#cur button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); state.currency = b.dataset.c;
  if (last) render(last, false);
});

/* ---------- api ---------- */
async function predict(){
  if (ctrl) ctrl.abort(); ctrl = new AbortController();
  $('#result').classList.add('loading');
  const body = {age:+$('#age').value, bmi:+$('#bmi').value, children:state.kids,
                sex:state.sex, smoker:state.smoker, region:state.region};
  try{
    const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'},
                                          body:JSON.stringify(body), signal:ctrl.signal});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Something went wrong.');
    last = d; render(d, true); $('#err').style.display = 'none';
  }catch(e){
    if (e.name === 'AbortError') return;
    const el = $('#err'); el.textContent = e.message; el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 4000);
  }finally{ $('#result').classList.remove('loading'); }
}

/* ---------- render ---------- */
function countUp(to){
  const from = shown, t0 = performance.now(), dur = 550;
  (function step(t){
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    const v = from + (to - from) * e;
    $('#estimate').textContent = money(v);
    $('#monthly').textContent = money(v / 12);
    if (k < 1) requestAnimationFrame(step); else shown = to;
  })(t0);
}

function render(d, animate){
  if (animate) countUp(d.estimate);
  else { shown = d.estimate; $('#estimate').textContent = money(d.estimate); $('#monthly').textContent = money(d.monthly); }

  const tier = $('#tier'); tier.textContent = d.tier + ' cost tier';
  tier.className = 'badge t-' + d.tier.split(' ')[0];
  $('#bmiBadge').textContent = 'BMI: ' + d.bmi_category;

  // meter 0 .. 50,000
  const MAX = 50000, pct = v => Math.min(98, Math.max(2, v / MAX * 100));
  $('#dot').style.left = pct(d.estimate) + '%';
  $('#avgMark').style.left = pct(d.baseline) + '%';
  $('#mMin').textContent = money(0); $('#mMax').textContent = money(MAX) + '+';
  const diff = d.estimate - d.baseline, rel = Math.abs(diff / d.baseline * 100);
  $('#vs').innerHTML = diff >= 0
    ? `That's <b class="up">${rel.toFixed(0)}% above</b> the average policyholder (${money(d.baseline)}).`
    : `That's <b class="down">${rel.toFixed(0)}% below</b> the average policyholder (${money(d.baseline)}).`;

  renderExplain(d); renderScen(d); drawChart(d);
}

function renderExplain(d){
  const card = $('#explainCard');
  if (!d.explain){ card.style.display = 'none'; return; }
  card.style.display = '';
  const max = Math.max(...d.explain.map(x => Math.abs(x.value)), 1);
  let html = `<div class="row base-row"><div class="name">Average policyholder</div><div></div><div class="val">${money(d.baseline)}</div></div>`;
  d.explain.forEach(x => {
    const w = Math.abs(x.value) / max * 50, pos = x.value >= 0;
    html += `<div class="row"><div class="name">${x.label}</div>
      <div class="track"><div class="fill ${pos ? 'pos' : 'neg'}" style="width:${w}%"></div></div>
      <div class="val ${pos ? 'pos' : 'neg'}">${signed(x.value)}</div></div>`;
  });
  html += `<div class="row" style="border-top:1px solid rgba(255,255,255,.18);margin-top:4px"><div class="name"><b>Your estimate</b></div><div></div><div class="val"><b>${money(d.estimate)}</b></div></div>`;
  $('#explain').innerHTML = html;
}

function renderScen(d){
  $('#scen').innerHTML = d.scenarios.map(s => {
    const save = s.delta < 0, abs = Math.abs(s.delta);
    const pill = abs < 1 ? '' :
      `<span class="pill ${save ? 'save' : 'more'}">${save ? 'Save ' + money(abs) : '+' + money(abs)} / yr</span>`;
    return `<div class="sc"><h3>${s.title}</h3><p>${s.desc}</p><div class="c">${money(s.cost)}</div>${pill}</div>`;
  }).join('');
}

function drawChart(d){
  const W = 640, H = 290, m = {l:62, r:16, t:16, b:36};
  const ages = d.curve.ages, a0 = ages[0], a1 = ages[ages.length - 1];
  const yMax = Math.max(5000, Math.ceil(Math.max(...d.curve.smoker, ...d.curve.nonsmoker) / 10000) * 10000);
  const x = a => m.l + (a - a0) / (a1 - a0) * (W - m.l - m.r);
  const y = v => H - m.b - v / yMax * (H - m.t - m.b);
  const path = arr => arr.map((v, i) => (i ? 'L' : 'M') + x(ages[i]).toFixed(1) + ' ' + y(v).toFixed(1)).join(' ');
  let g = '';
  for (let i = 0; i <= 4; i++){
    const v = yMax / 4 * i, yy = y(v);
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}" stroke="rgba(255,255,255,.08)"/>
          <text x="${m.l - 10}" y="${yy + 4}" text-anchor="end" fill="#93a0c0" font-size="11">${money(v)}</text>`;
  }
  for (let a = 20; a <= 60; a += 10)
    g += `<text x="${x(a)}" y="${H - 12}" text-anchor="middle" fill="#93a0c0" font-size="11">${a}</text>`;
  const age = +$('#age').value, ux = x(age), uy = y(d.estimate);
  $('#chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Estimated cost by age">
    ${g}
    <path d="${path(d.curve.smoker)} L${x(a1)} ${y(0)} L${x(a0)} ${y(0)} Z" fill="rgba(251,113,133,.08)"/>
    <path d="${path(d.curve.nonsmoker)} L${x(a1)} ${y(0)} L${x(a0)} ${y(0)} Z" fill="rgba(94,234,212,.08)"/>
    <path d="${path(d.curve.smoker)}" fill="none" stroke="#fb7185" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="${path(d.curve.nonsmoker)}" fill="none" stroke="#5eead4" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="${ux}" x2="${ux}" y1="${m.t}" y2="${H - m.b}" stroke="rgba(255,255,255,.35)" stroke-dasharray="4 4"/>
    <circle cx="${ux}" cy="${uy}" r="8" fill="#fff" stroke="#0b1020" stroke-width="4"/>
    <text x="${Math.min(ux + 14, W - 90)}" y="${Math.max(uy - 14, 14)}" fill="#fff" font-size="12" font-weight="700">${money(d.estimate)}</text>
  </svg>`;
}

syncLabels();
predict();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
