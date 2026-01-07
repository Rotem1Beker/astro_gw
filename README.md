# 🌊 Gravitational Wave Detection with Neural Posterior Estimation

## סקירה כללית

פרויקט זה משתמש ב-**Normalizing Flows** ו-**Neural Posterior Estimation** לזיהוי ואפיון גלי כבידה מנתוני LIGO. המודל לומד להעריך את ההתפלגות הפוסטריורית של פרמטרי המקור (מסות, מרחק, וכו') ישירות מהאות הנמדד.

---

## 📁 מבנה הפרויקט

```
astro_gw/
├── src/
│   ├── model/
│   │   ├── flow.py              # Normalizing Flow architecture
│   │   └── conditioning.py      # Context encoder
│   ├── preprocess.py            # Signal preprocessing (FFT, whitening)
│   ├── train.py                 # Original training script
│   ├── infer_online.py          # Inference on real LIGO data
│   │
│   ├── analyze_event_window.py      # 🆕 Event vs noise analysis
│   ├── bayes_factor.py              # 🆕 Bayesian model comparison
│   ├── train_realistic_waveforms.py # 🆕 Training with chirp signals
│   └── run_full_research_pipeline.py # 🆕 Master orchestration script
│
├── data/
│   └── H-H1_LOSC_4_V1-1126259446-32.hdf5  # GW150914 data
├── config.yaml
├── model.pt                     # Trained model (toy waveforms)
└── model_realistic.pt           # Trained model (realistic chirps)
```

---

## 🚀 פייפליין המחקר

### שלב 1: ניתוח חלון האירוע 🔍

```bash
python analyze_event_window.py
```

**מה זה עושה:**
- מבודד חלון ±0.2s סביב זמן האירוע GW150914
- משווה לחלון רעש רקע (מתחילת הקובץ)
- מריץ inference על שני החלונות
- מייצר:
  - Time series וספקטרוגרמות
  - התפלגויות פוסטריוריות (2D scatter plots)
  - Bayes Factor פשוט

**פלט:** `event_vs_noise_analysis.png`

---

### שלב 2: Bayes Factor מפורט 📊

```bash
python bayes_factor.py
```

**מה זה עושה:**
- מחשב Bayes Factor בשלוש שיטות:
  1. **Importance Sampling** - מעריך evidence ישירות
  2. **Gaussian Approximation** - מניח פוסטריור גאוסי
  3. **KL Divergence** - מודד שוני בין posteriors
- מפרש תוצאות לפי Jeffrey's Scale:
  - log BF > 5: Strong evidence for signal
  - log BF > 3: Substantial evidence
  - log BF > 1: Weak evidence
  - -1 < log BF < 1: Inconclusive

**פלט:** טקסט עם פירוש מפורט

---

### שלב 3: Waveforms ריאליסטיים 🌊

```bash
python train_realistic_waveforms.py
```

**מה זה עושה:**
- מייצר **chirp signals** ריאליסטיים:
  ```python
  f(t) ∝ (t_to_merger)^(-3/8)  # תדר עולה
  h(t) ∝ f(t)^(2/3)            # אמפליטודה עולה
  ```
- מוסיף רעש צבעוני (LIGO-like PSD)
- מאמן מודל חדש על 10,000 דוגמאות
- פרמטרים:
  - m1, m2: 5-50 M☉
  - distance: 100-1000 Mpc
  - inclination: 0-π

**פלט:** 
- `realistic_waveforms_test.png` - דוגמאות waveforms
- `model_realistic.pt` - מודל מאומן חדש
- `training_loss_realistic.png` - עקומת אימון

---

### שלב 4: Inference עם מודל ריאליסטי 🎯

```bash
python infer_online.py --model model_realistic.pt
```

**מה זה עושה:**
- מריץ inference על GW150914 עם המודל החדש
- משווה תוצאות למודל המקורי
- אמור לתת הערכות מסות קרובות יותר ל:
  - m1 ≈ 36 M☉
  - m2 ≈ 29 M☉

---

## 🎬 הרצת פייפליין מלא

### מצב אינטראקטיבי:
```bash
python run_full_research_pipeline.py
```

מציג תפריט עם כל השלבים, מאפשר לבחור צעד ספציפי או להריץ הכל.

### מצב אוטומטי:
```bash
python run_full_research_pipeline.py --step all
```

מריץ את כל הצעדים ברצף ללא התערבות.

---

## 📈 תוצאות צפויות

### מודל Toy (קיים):
- ✅ **עובד**: מייצר posterior distribution
- ❌ **לא ריאליסטי**: לא מתאים לפיזיקה אמיתית
- התוצאות: פרמטרים אקראיים ללא משמעות פיזיקלית

### מודל Realistic (חדש):
- ✅ **ריאליסטי**: מאומן על chirp signals אמיתיים
- ✅ **פיזיקלי**: מעריך מסות, מרחק, זווית
- תוצאות צפויות על GW150914:
  - m1: 30-40 M☉ (אמת: 36)
  - m2: 25-35 M☉ (אמת: 29)
  - uncertainty תלוי ב-SNR ובאיכות האימון

### Bayes Factor:
- Signal vs Noise על חלון האירוע:
  - צפוי: **log BF > 5** (strong evidence)
- Noise vs Noise על רעש רקע:
  - צפוי: **log BF ≈ 0** (inconclusive)

---

## 🔬 שיפורים עתידיים

### קצר טווח:
1. ✅ חלון זמן קצר סביב אירוע
2. ✅ השוואה לרעש רקע
3. ✅ Bayes Factor
4. ✅ אימון על waveforms ריאליסטיים

### בינוני:
5. 🔄 הוספת פרמטרים: distance, inclination, spin
6. 🔄 אימון על מספר אירועים (transfer learning)
7. 🔄 ניתוח רגישות: SNR, duration, preprocessing

### ארוך טווח:
8. 📋 אימון על waveforms מדויקים (PyCBC, LALSuite)
9. 📋 Multiple detectors (H1, L1, V1)
10. 📋 Real-time detection pipeline
11. 📋 השוואה ל-LALInference (golden standard)

---

## 🛠️ דרישות

```bash
pip install torch numpy scipy matplotlib h5py pyyaml tqdm
```

**קבצים נדרשים:**
- `data/H-H1_LOSC_4_V1-1126259446-32.hdf5` - נתוני LIGO
- `config.yaml` - קונפיגורציה
- `model.pt` - מודל מאומן ראשוני

---

## 📚 רקע תיאורטי

### Neural Posterior Estimation:
במקום לחשב `p(θ|x)` באמצעות MCMC (איטי), אנו לומדים:
```
q_φ(θ|x) ≈ p(θ|x)
```
באמצעות Normalizing Flow שמקבל את האות `x` כ-conditioning.

### Normalizing Flow:
טרנספורמציה הפיכה:
```
θ = f_φ(z, x)    where z ~ N(0,I)
p(θ|x) = p(z) |det J_f^(-1)|
```

### Bayes Factor:
```
BF = P(data | signal) / P(data | noise)
log BF > 5  → strong evidence for signal
```

---

## 📖 מקורות

1. Lueckmann et al. (2021) - "Likelihood-free inference with neural posterior estimation"
2. Abbott et al. (2016) - "Observation of Gravitational Waves from GW150914"
3. LIGO Open Science Center - https://gwosc.org
4. PyCBC/LALSuite - Standard GW analysis tools

---

## 👩‍🔬 שימוש מחקרי

פרויקט זה הוא **proof of concept** להשוואה בין:
- NPE (fast, approximate)
- MCMC/LALInference (slow, exact)

**לא** מיועד לשימוש מדעי רציני ללא:
- Validation מול pipeline סטנדרטי
- אימון על נתונים גדולים יותר
- כיול פיזיקלי מדויק

---

## 🤝 תרומה

Pull requests מתקבלים בברכה! תחומים מעניינים:
- שיפור generator של waveforms
- הוספת פרמטרים (spins, eccentricity)
- ויזואליזציות נוספות
- אופטימיזציה של האימון

---

## 📄 רישיון

MIT License - ראה LICENSE לפרטים

---

## 🙏 תודות

- LIGO Scientific Collaboration - על הנתונים הפתוחים
- Anthropic Claude - על הסיוע בפיתוח
- קהילת ML4Science - על השראה והכוונה