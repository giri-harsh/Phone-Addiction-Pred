# 📱 Phone Addiction Predictor

> **Can we predict teen smartphone addiction from usage patterns?**  
> *Spoiler: Yes, with 84% accuracy* 🎯

## 🚀 What This Does

Built a machine learning model that predicts teen smartphone addiction levels (1-10 scale) using behavioral data. Turns out phone usage, sleep patterns, and mental health create a pretty clear picture of addiction risk.

**The bottom line:** MAE of 0.38 means we're predicting within ~0.4 points on a 10-point scale. Not bad for reading someone's digital soul.

## 🔥 Key Findings

```
📊 Model Performance: 84% accuracy (R² = 0.84)
🎮 Gaming addicts sleep 2+ hours less than casual users
📚 Heavy phone users score 15-20% lower academically  
😰 Anxiety levels spike with addiction scores
🌙 Bedtime scrollers = worse sleep quality
```

## 🛠️ What's Under the Hood

**Data Preprocessing Magic:**
- IQR outlier detection and removal (because some people claim 25hr/day usage 🙄)
- One-Hot Encoded categorical features (Gender, School Grade, Usage Purpose)
- StandardScaler normalization for the Random Forest

**The Analysis Pipeline:**
```python
Raw Data → Outlier Removal → Feature Engineering → EDA → ML Model → Insights
```

**Model:** Random Forest Regressor (100 estimators, because why not)


## 📈 The Numbers

| Metric | Score | What This Means |
|--------|-------|----------------|
| **R² Score** | 0.84 | Model explains 84% of addiction variance |
| **MAE** | 0.38 | Average prediction error is ~0.4 points |
| **MSE** | 0.25 | Low squared error = consistent predictions |

## 🏃‍♂️ Quick Start

```bash
git clone https://github.com/giri-harsh/Phone-Addiction-Pred.git
cd Phone-Addiction-Pred
pip install pandas numpy scikit-learn matplotlib seaborn
python code3.py  # Run the full analysis
```

All visualizations will be displayed as the script runs.

## 🔍 What I Discovered

**Top Addiction Predictors:**
1. Daily usage hours (obviously)
2. Sleep quality degradation  
3. Gaming time vs academic performance
4. Mental health indicators (anxiety/depression)
5. Bedtime phone usage patterns

**Surprising Insights:**
- Social media usage correlates stronger with addiction than gaming
- Self-esteem drops significantly at addiction levels 7+
- Academic performance has a sharp cliff, not gradual decline

## 🚀 What's Next

- **Streamlit Dashboard:** Interactive prediction tool
- **Real-time Monitoring:** Phone app integration  
- **Classification Model:** Low/Medium/High risk categories
- **Intervention Recommendations:** Personalized digital wellness tips

## 👨‍💻 Built By

**Harsh Giri** | CS @ AKGEC  
*Turning data into insights, one dataset at a time*

Got questions? Found a bug? Want to collaborate? Drop an issue or connect!

---

*⚡ 84% accuracy in predicting digital addiction - your phone usage patterns are more predictable than you think*

*⚡ 84% accuracy in predicting digital addiction - your phone usage patterns are more predictable than you think*
