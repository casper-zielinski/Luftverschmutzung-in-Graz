# Main.py – Aufgaben 1–3 (Final Version)
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# -------------------------------------------------------------
# Ordner für Plots anlegen
# -------------------------------------------------------------
if not os.path.exists("plots"):
    os.makedirs("plots")

# -------------------------------------------------------------
# 1. Daten laden
# -------------------------------------------------------------
with open("feinstaubdataexercise.pickle", "rb") as file:
    dailymeansdata = pkl.load(file)

graz = dailymeansdata["Graz-DB"]
kalk = dailymeansdata["Kalkleiten"]

# Index in Datum umwandeln
graz.index = pd.to_datetime(graz.index)
kalk.index = pd.to_datetime(kalk.index)

# Trainingsdaten (2015–2019)
graz_train = graz.loc["2015":"2019"].copy()
kalk_train = kalk.loc["2015":"2019"].copy()

print("--- Datensatz Überblick ---")
print(graz_train.describe())
print("\nMissing Values:")
print(graz_train.isna().sum())

# -------------------------------------------------------------
# 1. Grafiken speichern (Aufgabe 1)
# -------------------------------------------------------------

# Zeitverlauf PM10 & NO2
plt.figure(figsize=(12, 6))
plt.plot(graz_train.index, graz_train["pm10"], label="PM10")
plt.plot(graz_train.index, graz_train["no2"], label="NO2")
plt.title("Zeitverlauf von PM10 und NO2 (2015–2019)")
plt.xlabel("Datum")
plt.ylabel("µg/m³")
plt.legend()
plt.tight_layout()
plt.savefig("plots/A1_zeitverlauf_pm10_no2.png")
plt.close()

# Scatterplots
predictors = ["humidity", "temp", "prec", "windspeed", "peak_velocity"]

for col in predictors:
    # PM10
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=graz_train[col], y=graz_train["pm10"], alpha=0.5)
    plt.title(f"PM10 vs {col}")
    plt.tight_layout()
    plt.savefig(f"plots/A1_pm10_vs_{col}.png")
    plt.close()

    # NO2
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=graz_train[col], y=graz_train["no2"], alpha=0.5)
    plt.title(f"NO2 vs {col}")
    plt.tight_layout()
    plt.savefig(f"plots/A1_no2_vs_{col}.png")
    plt.close()

# -------------------------------------------------------------
# 2. Erste Modelle für PM10 und NO2 (Aufgabe 2)
# -------------------------------------------------------------

formula_pm10 = "pm10 ~ humidity + temp + prec + windspeed + peak_velocity + C(day_type)"
formula_no2 = "no2 ~ humidity + temp + prec + windspeed + peak_velocity + C(day_type)"

model_pm10 = ols(formula_pm10, data=graz_train).fit()
model_no2 = ols(formula_no2, data=graz_train).fit()

print("\n--- PM10 Modell Summary ---")
print(model_pm10.summary())

print("\n--- NO2 Modell Summary ---")
print(model_no2.summary())

print("\n--- ANOVA PM10 ---")
print(sm.stats.anova_lm(model_pm10))

print("\n--- ANOVA NO2 ---")
print(sm.stats.anova_lm(model_no2))

# -------------------------------------------------------------
# 3. Inversion: Temperaturdifferenz Graz - Kalkleiten (Aufgabe 3)
# -------------------------------------------------------------

merged = pd.merge(
    graz_train,
    kalk_train[["temp"]],
    how="inner",
    left_index=True,
    right_index=True,
    suffixes=("", "_kalk")
)

# Temperaturdifferenz: Graz – Kalkleiten
merged["temp_diff"] = merged["temp"] - merged["temp_kalk"]

# Plot temp_diff
plt.figure(figsize=(10, 4))
plt.plot(merged.index, merged["temp_diff"])
plt.axhline(0, color="red", linestyle="--")
plt.title("Temperaturdifferenz Graz - Kalkleiten (Inversion < 0)")
plt.ylabel("°C")
plt.tight_layout()
plt.savefig("plots/A3_temp_diff_graz_kalkleiten.png")
plt.close()

# Modelle inkl. temp_diff
formula_pm10_inv = "pm10 ~ humidity + temp + prec + windspeed + peak_velocity + temp_diff + C(day_type)"
formula_no2_inv = "no2 ~ humidity + temp + prec + windspeed + peak_velocity + temp_diff + C(day_type)"

model_pm10_inv = ols(formula_pm10_inv, data=merged).fit()
model_no2_inv = ols(formula_no2_inv, data=merged).fit()

print("\n--- PM10 Modell mit temp_diff ---")
print(model_pm10_inv.summary())

print("\n--- NO2 Modell mit temp_diff ---")
print(model_no2_inv.summary())

# -------------------------------------------------------------
# 4. Feature Enginnering (Aufgabe 4)
# -------------------------------------------------------------

# mit merged dataframe weiterarbeiten
data = merged.copy()

# Frost-Indikator: Temperatur unter 0°C
data["frost"] = (data["temp"] < 0).astype(int)

# Inversion Indikator: Wenn Graz kälter als Kalkleiten
data["inversion"] = (data["temp_diff"] < 0).astype(int)

# Strong Wind Indikator: Wind Geschwindigkeit schneller als 0.6 m/s
data["strong_wind"] = (data["windspeed"] < 0.6).astype(int)

# Starker Regen: Niederschlag über 5 l/m²
data["heavy_rain"] = (data["prec"] > 5).astype(int)

# Jahr als kategorische Variable (für Trends über Jahre)
data["year"] = data.index.year

# Monat (für saisonale Effekte)
data["month"] = data.index.month

# Quartal/Saison
data["season"] = data.index.quarter  # 1=Winter, 2=Frühjahr, 3=Sommer, 4=Herbst

# Wochentag als Zahl (0=Montag, 6=Sonntag)
data["weekday"] = data.index.dayofweek

print(f"\nBinäre Features erstellt: frost, inversion, strong_wind, heavy_rain")
print(f"Zeitliche Features erstellt: year, month, season, weekday")

# Statistik der neuen binären Features
print("\n--- Häufigkeit der binären Features ---")
print(f"Frosttage: {data['frost'].sum()} ({data['frost'].mean()*100:.1f}%)")
print(f"Inversionstage: {data['inversion'].sum()} ({data['inversion'].mean()*100:.1f}%)")
print(f"Starker Wind: {data['strong_wind'].sum()} ({data['strong_wind'].mean()*100:.1f}%)")
print(f"Starker Regen: {data['heavy_rain'].sum()} ({data['heavy_rain'].mean()*100:.1f}%)")

# -------------------------------------------------------------
# 4.2 Lagged Variables (Vortageswerte)
# -------------------------------------------------------------

# Meteorologische Variablen vom Vortag
data["temp_lag1"] = data["temp"].shift(1)
data["humidity_lag1"] = data["humidity"].shift(1)
data["windspeed_lag1"] = data["windspeed"].shift(1)
data["prec_lag1"] = data["prec"].shift(1)

# Schadstoffwerte vom Vortag
data["pm10_lag1"] = data["pm10"].shift(1)
data["no2_lag1"] = data["no2"].shift(1)

# 2-Tages-Lag
data["pm10_lag2"] = data["pm10"].shift(2)
data["no2_lag2"] = data["no2"].shift(2)

# NaN-Werte durch Shift entfernen
rows_before = len(data)
data = data.dropna()
rows_after = len(data)
print(f"Lagged Variables erstellt, {rows_before - rows_after} Zeilen mit NaN entfernt")

# -------------------------------------------------------------
# 4.3 Visualisierung: Effekt der neuen Features
# -------------------------------------------------------------

# Plot: PM10 bei Frost vs. kein Frost
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].boxplot([data[data["frost"]==0]["pm10"], data[data["frost"]==1]["pm10"]], 
                tick_labels=["Kein Frost", "Frost"])
axes[0].set_ylabel("PM10 (µg/m³)")
axes[0].set_title("PM10: Frost vs. kein Frost")

axes[1].boxplot([data[data["inversion"]==0]["pm10"], data[data["inversion"]==1]["pm10"]], 
                tick_labels=["Keine Inversion", "Inversion"])
axes[1].set_ylabel("PM10 (µg/m³)")
axes[1].set_title("PM10: Inversion vs. keine Inversion")

plt.tight_layout()
plt.savefig("plots/A4_frost_inversion_effect.png")
plt.close()

# Korrelation zwischen pm10 heute und gestern
plt.figure(figsize=(6, 5))
sns.scatterplot(x=data["pm10_lag1"], y=data["pm10"], alpha=0.5)
plt.plot([0, 100], [0, 100], 'r--', label='y=x')
plt.xlabel("PM10 gestern (µg/m³)")
plt.ylabel("PM10 heute (µg/m³)")
plt.title("Autokorrelation PM10 (heute vs. gestern)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/A4_pm10_autocorrelation.png")
plt.close()

# -------------------------------------------------------------
# 4.4 Modelle mit neuen Features
# -------------------------------------------------------------

# Erweiterte Formel für PM10
# year OHNE C() -> numerisch
formula_pm10_v3 = """pm10 ~ humidity + temp + prec + windspeed + peak_velocity + 
                     temp_diff + frost + inversion + strong_wind + heavy_rain +
                     year + C(day_type) + 
                     temp_lag1 + humidity_lag1 + windspeed_lag1 + pm10_lag1"""

formula_no2_v3 = """no2 ~ humidity + temp + prec + windspeed + peak_velocity + 
                    temp_diff + frost + inversion + strong_wind + heavy_rain +
                    year + C(day_type) + 
                    temp_lag1 + humidity_lag1 + windspeed_lag1 + no2_lag1"""

model_pm10_v3 = ols(formula_pm10_v3, data=data).fit()
model_no2_v3 = ols(formula_no2_v3, data=data).fit()

print("\n--- PM10 Modell v3 (mit Feature Engineering) ---")
print(f"R² = {model_pm10_v3.rsquared:.4f}, Adj. R² = {model_pm10_v3.rsquared_adj:.4f}")

print("\n--- NO2 Modell v3 (mit Feature Engineering) ---")
print(f"R² = {model_no2_v3.rsquared:.4f}, Adj. R² = {model_no2_v3.rsquared_adj:.4f}")

# Vergleich mit altem Modell
model_pm10_old = ols(formula_pm10_inv, data=data).fit()
model_no2_old = ols(formula_no2_inv, data=data).fit()

pm10_improvement = (model_pm10_v3.rsquared - model_pm10_old.rsquared) / model_pm10_old.rsquared * 100
no2_improvement = (model_no2_v3.rsquared - model_no2_old.rsquared) / model_no2_old.rsquared * 100

print(f"\nVerbesserung PM10: R² um {pm10_improvement:.2f}% gestiegen")
print(f"Verbesserung NO2: R² um {no2_improvement:.2f}% gestiegen")

# Feature Importance
pm10_pvalues = model_pm10_v3.pvalues
pm10_significant = pm10_pvalues[pm10_pvalues < 0.05].sort_values()

print("\nPM10 - Top 10 signifikante Features:")
for feature, pval in pm10_significant.head(10).items():
    coef = model_pm10_v3.params[feature]
    print(f"  {feature:25s}: p={pval:.4f}, coef={coef:+.4f}")

# Diagnostische Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(model_pm10_v3.fittedvalues, model_pm10_v3.resid, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('PM10: Residuals vs Fitted')

sm.qqplot(model_pm10_v3.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('PM10: Q-Q Plot')

axes[1, 0].scatter(model_no2_v3.fittedvalues, model_no2_v3.resid, alpha=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('NO2: Residuals vs Fitted')

sm.qqplot(model_no2_v3.resid, line='45', ax=axes[1, 1])
axes[1, 1].set_title('NO2: Q-Q Plot')

plt.tight_layout()
plt.savefig("plots/A4_diagnostics_v3.png")
plt.close()

# -------------------------------------------------------------
# 5. Response-Transformation (Aufgabe 5)
# -------------------------------------------------------------

print("\n--- Aufgabe 5: Response-Transformation ---")

data["pm10_sqrt"] = np.sqrt(data["pm10"])
data["no2_sqrt"] = np.sqrt(data["no2"])

# year OHNE C()
formula_pm10_sqrt = """pm10_sqrt ~ humidity + temp + prec + windspeed + peak_velocity + 
                       temp_diff + frost + inversion + strong_wind + heavy_rain +
                       year + C(day_type) + 
                       temp_lag1 + humidity_lag1 + windspeed_lag1 + pm10_lag1"""

formula_no2_sqrt = """no2_sqrt ~ humidity + temp + prec + windspeed + peak_velocity + 
                      temp_diff + frost + inversion + strong_wind + heavy_rain +
                      year + C(day_type) + 
                      temp_lag1 + humidity_lag1 + windspeed_lag1 + no2_lag1"""

model_pm10_sqrt = ols(formula_pm10_sqrt, data=data).fit()
model_no2_sqrt = ols(formula_no2_sqrt, data=data).fit()

print(f"PM10 transformiert: R² = {model_pm10_sqrt.rsquared:.4f}, Adj. R² = {model_pm10_sqrt.rsquared_adj:.4f}")
print(f"NO2 transformiert:  R² = {model_no2_sqrt.rsquared:.4f}, Adj. R² = {model_no2_sqrt.rsquared_adj:.4f}")

print("\nVergleich der Modelle:")
print(f"PM10 - Original R²: {model_pm10_v3.rsquared:.4f}, Transformiert R²: {model_pm10_sqrt.rsquared:.4f}")
print(f"NO2  - Original R²: {model_no2_v3.rsquared:.4f}, Transformiert R²: {model_no2_sqrt.rsquared:.4f}")

# Diagnostische Plots: Vorher vs. Nachher
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Diagnostik: Original vs. Transformiert', fontsize=14)

# PM10 Original
axes[0, 0].scatter(model_pm10_v3.fittedvalues, model_pm10_v3.resid, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('PM10 Original: Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted')
axes[0, 0].set_ylabel('Residuals')

sm.qqplot(model_pm10_v3.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('PM10 Original: Q-Q Plot')

# PM10 Transformiert
axes[0, 2].scatter(model_pm10_sqrt.fittedvalues, model_pm10_sqrt.resid, alpha=0.5)
axes[0, 2].axhline(y=0, color='r', linestyle='--')
axes[0, 2].set_title('PM10 √-Transform: Residuals vs Fitted')
axes[0, 2].set_xlabel('Fitted')
axes[0, 2].set_ylabel('Residuals')

sm.qqplot(model_pm10_sqrt.resid, line='45', ax=axes[0, 3])
axes[0, 3].set_title('PM10 √-Transform: Q-Q Plot')

# NO2 Original
axes[1, 0].scatter(model_no2_v3.fittedvalues, model_no2_v3.resid, alpha=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('NO2 Original: Residuals vs Fitted')
axes[1, 0].set_xlabel('Fitted')
axes[1, 0].set_ylabel('Residuals')

sm.qqplot(model_no2_v3.resid, line='45', ax=axes[1, 1])
axes[1, 1].set_title('NO2 Original: Q-Q Plot')

# NO2 Transformiert
axes[1, 2].scatter(model_no2_sqrt.fittedvalues, model_no2_sqrt.resid, alpha=0.5)
axes[1, 2].axhline(y=0, color='r', linestyle='--')
axes[1, 2].set_title('NO2 √-Transform: Residuals vs Fitted')
axes[1, 2].set_xlabel('Fitted')
axes[1, 2].set_ylabel('Residuals')

sm.qqplot(model_no2_sqrt.resid, line='45', ax=axes[1, 3])
axes[1, 3].set_title('NO2 √-Transform: Q-Q Plot')

plt.tight_layout()
plt.savefig("plots/A5_transformation_comparison.png", dpi=150)
plt.close()

# Entscheidung: Welches Modell für Vorhersage?
use_transform_pm10 = model_pm10_sqrt.rsquared > model_pm10_v3.rsquared
use_transform_no2 = model_no2_sqrt.rsquared > model_no2_v3.rsquared

print(f"\nFür Vorhersage verwenden:")
print(f"PM10: {'Transformiertes' if use_transform_pm10 else 'Original'} Modell")
print(f"NO2:  {'Transformiertes' if use_transform_no2 else 'Original'} Modell")

final_model_pm10 = model_pm10_sqrt if use_transform_pm10 else model_pm10_v3
final_model_no2 = model_no2_sqrt if use_transform_no2 else model_no2_v3

# -------------------------------------------------------------
# 6. Vorhersage für 2020 (Aufgabe 6)
# -------------------------------------------------------------

print("\n--- Aufgabe 6: Vorhersage für 2020 ---")

graz_2020 = graz.loc["2020"].copy()
kalk_2020 = kalk.loc["2020"].copy()

test_2020 = pd.merge(
    graz_2020,
    kalk_2020[["temp"]],
    how="inner",
    left_index=True,
    right_index=True,
    suffixes=("", "_kalk")
)

# Alle Features erstellen (wie in Training)
test_2020["temp_diff"] = test_2020["temp"] - test_2020["temp_kalk"]
test_2020["frost"] = (test_2020["temp"] < 0).astype(int)
test_2020["inversion"] = (test_2020["temp_diff"] < 0).astype(int)
test_2020["strong_wind"] = (test_2020["windspeed"] > 0.6).astype(int)
test_2020["heavy_rain"] = (test_2020["prec"] > 5).astype(int)
test_2020["year"] = test_2020.index.year  # Kann jetzt 2020 sein!
test_2020["month"] = test_2020.index.month
test_2020["season"] = test_2020.index.quarter
test_2020["weekday"] = test_2020.index.dayofweek

# Lagged variables
last_day_2019 = data.iloc[-1]

test_2020["temp_lag1"] = np.nan
test_2020["humidity_lag1"] = np.nan
test_2020["windspeed_lag1"] = np.nan
test_2020["prec_lag1"] = np.nan
test_2020["pm10_lag1"] = np.nan
test_2020["no2_lag1"] = np.nan

# Erster Tag 2020: Nutze letzten Tag 2019
test_2020.iloc[0, test_2020.columns.get_loc("temp_lag1")] = last_day_2019["temp"]
test_2020.iloc[0, test_2020.columns.get_loc("humidity_lag1")] = last_day_2019["humidity"]
test_2020.iloc[0, test_2020.columns.get_loc("windspeed_lag1")] = last_day_2019["windspeed"]
test_2020.iloc[0, test_2020.columns.get_loc("prec_lag1")] = last_day_2019["prec"]
test_2020.iloc[0, test_2020.columns.get_loc("pm10_lag1")] = last_day_2019["pm10"]
test_2020.iloc[0, test_2020.columns.get_loc("no2_lag1")] = last_day_2019["no2"]

# Restliche Tage: Nutze echte Werte vom Vortag
for i in range(1, len(test_2020)):
    test_2020.iloc[i, test_2020.columns.get_loc("temp_lag1")] = test_2020.iloc[i-1]["temp"]
    test_2020.iloc[i, test_2020.columns.get_loc("humidity_lag1")] = test_2020.iloc[i-1]["humidity"]
    test_2020.iloc[i, test_2020.columns.get_loc("windspeed_lag1")] = test_2020.iloc[i-1]["windspeed"]
    test_2020.iloc[i, test_2020.columns.get_loc("prec_lag1")] = test_2020.iloc[i-1]["prec"]
    test_2020.iloc[i, test_2020.columns.get_loc("pm10_lag1")] = test_2020.iloc[i-1]["pm10"]
    test_2020.iloc[i, test_2020.columns.get_loc("no2_lag1")] = test_2020.iloc[i-1]["no2"]

# Vorhersagen machen
if use_transform_pm10:
    test_2020["pm10_pred_sqrt"] = final_model_pm10.predict(test_2020)
    test_2020["pm10_pred"] = test_2020["pm10_pred_sqrt"] ** 2
else:
    test_2020["pm10_pred"] = final_model_pm10.predict(test_2020)

if use_transform_no2:
    test_2020["no2_pred_sqrt"] = final_model_no2.predict(test_2020)
    test_2020["no2_pred"] = test_2020["no2_pred_sqrt"] ** 2
else:
    test_2020["no2_pred"] = final_model_no2.predict(test_2020)

# NaN-Werte entfernen vor Evaluation
print(f"\nVor dropna: {len(test_2020)} Zeilen")
print(f"NaN in pm10: {test_2020['pm10'].isna().sum()}")
print(f"NaN in pm10_pred: {test_2020['pm10_pred'].isna().sum()}")
print(f"NaN in no2: {test_2020['no2'].isna().sum()}")
print(f"NaN in no2_pred: {test_2020['no2_pred'].isna().sum()}")

# Entferne Zeilen mit NaN in den relevanten Spalten
test_2020_clean = test_2020.dropna(subset=["pm10", "pm10_pred", "no2", "no2_pred"])
print(f"Nach dropna: {len(test_2020_clean)} Zeilen")

# Evaluationsmetriken
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pm10_rmse = np.sqrt(mean_squared_error(test_2020_clean["pm10"], test_2020_clean["pm10_pred"]))
pm10_mae = mean_absolute_error(test_2020_clean["pm10"], test_2020_clean["pm10_pred"])
pm10_r2 = r2_score(test_2020_clean["pm10"], test_2020_clean["pm10_pred"])

no2_rmse = np.sqrt(mean_squared_error(test_2020_clean["no2"], test_2020_clean["no2_pred"]))
no2_mae = mean_absolute_error(test_2020_clean["no2"], test_2020_clean["no2_pred"])
no2_r2 = r2_score(test_2020_clean["no2"], test_2020_clean["no2_pred"])

print(f"\nPM10 Vorhersage-Performance:")
print(f"  RMSE: {pm10_rmse:.2f} µg/m³")
print(f"  MAE:  {pm10_mae:.2f} µg/m³")
print(f"  R²:   {pm10_r2:.4f}")

print(f"\nNO2 Vorhersage-Performance:")
print(f"  RMSE: {no2_rmse:.2f} µg/m³")
print(f"  MAE:  {no2_mae:.2f} µg/m³")
print(f"  R²:   {no2_r2:.4f}")

# -------------------------------------------------------------
# 7. COVID-19 Vergleich (Aufgabe 7)
# -------------------------------------------------------------

print("\n--- Aufgabe 7: COVID-19 Analyse ---")

# Verwende test_2020_clean für die Plots (ohne NaN)
test_6m = test_2020_clean[test_2020_clean.index < "2020-07-01"]

# COVID-19 Lockdown Daten für Österreich
# Prüfe, ob Index timezone-aware ist
if test_6m.index.tz is not None:
    # Index ist timezone-aware -> Timestamps auch timezone-aware machen
    lockdown_start = pd.Timestamp("2020-03-16", tz=test_6m.index.tz)
    lockdown_end = pd.Timestamp("2020-05-01", tz=test_6m.index.tz)
    lockdown_end2 = pd.Timestamp("2020-05-15", tz=test_6m.index.tz)
else:
    # Index ist timezone-naive -> normale Timestamps
    lockdown_start = pd.Timestamp("2020-03-16")
    lockdown_end = pd.Timestamp("2020-05-01")
    lockdown_end2 = pd.Timestamp("2020-05-15")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# PM10
axes[0].plot(test_6m.index, test_6m["pm10"], 
             label="Tatsächliche Werte", color="blue", linewidth=2, marker='o', markersize=3)
axes[0].plot(test_6m.index, test_6m["pm10_pred"], 
             label="Vorhersage (Modell)", color="red", linestyle="--", linewidth=2)
axes[0].axvline(lockdown_start, color="green", linestyle=":", linewidth=2, label="Lockdown Start (16. März)")
axes[0].axvline(lockdown_end, color="orange", linestyle=":", linewidth=2, label="1. Lockerungen (1. Mai)")
axes[0].axvline(lockdown_end2, color="purple", linestyle=":", linewidth=2, label="2. Lockerungen (15. Mai)")
axes[0].axvspan(lockdown_start, lockdown_end2, alpha=0.2, color="gray", label="Lockdown-Phase")
axes[0].set_xlabel("Datum", fontsize=11)
axes[0].set_ylabel("PM10 (µg/m³)", fontsize=11)
axes[0].set_title("PM10: Vorhersage vs. Realität 2020 (mit COVID-19 Lockdown)", fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)

# NO2
axes[1].plot(test_6m.index, test_6m["no2"], 
             label="Tatsächliche Werte", color="blue", linewidth=2, marker='o', markersize=3)
axes[1].plot(test_6m.index, test_6m["no2_pred"], 
             label="Vorhersage (Modell)", color="red", linestyle="--", linewidth=2)
axes[1].axvline(lockdown_start, color="green", linestyle=":", linewidth=2, label="Lockdown Start (16. März)")
axes[1].axvline(lockdown_end, color="orange", linestyle=":", linewidth=2, label="1. Lockerungen (1. Mai)")
axes[1].axvline(lockdown_end2, color="purple", linestyle=":", linewidth=2, label="2. Lockerungen (15. Mai)")
axes[1].axvspan(lockdown_start, lockdown_end2, alpha=0.2, color="gray", label="Lockdown-Phase")
axes[1].set_xlabel("Datum", fontsize=11)
axes[1].set_ylabel("NO2 (µg/m³)", fontsize=11)
axes[1].set_title("NO2: Vorhersage vs. Realität 2020 (mit COVID-19 Lockdown)", fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/A7_covid19_comparison_6months.png", dpi=150)
plt.close()

# Analyse
pre_lockdown = test_6m[test_6m.index < lockdown_start]
during_lockdown = test_6m[(test_6m.index >= lockdown_start) & (test_6m.index <= lockdown_end2)]
after_lockdown = test_6m[test_6m.index > lockdown_end2]

print("\nDurchschnittliche PM10-Werte:")
print(f"  Vor Lockdown (Jan-15.März):      Tatsächlich: {pre_lockdown['pm10'].mean():.1f}, Vorhergesagt: {pre_lockdown['pm10_pred'].mean():.1f}")
print(f"  Während Lockdown (16.März-15.Mai): Tatsächlich: {during_lockdown['pm10'].mean():.1f}, Vorhergesagt: {during_lockdown['pm10_pred'].mean():.1f}")
print(f"  Nach Lockdown (16.Mai-30.Juni):    Tatsächlich: {after_lockdown['pm10'].mean():.1f}, Vorhergesagt: {after_lockdown['pm10_pred'].mean():.1f}")

pm10_lockdown_diff = during_lockdown['pm10'].mean() - during_lockdown['pm10_pred'].mean()
no2_lockdown_diff = during_lockdown['no2'].mean() - during_lockdown['no2_pred'].mean()

print(f"\nAbweichung während Lockdown:")
print(f"  PM10: {pm10_lockdown_diff:.1f} µg/m³ niedriger als erwartet ({pm10_lockdown_diff/during_lockdown['pm10_pred'].mean()*100:.1f}%)")
print(f"  NO2:  {no2_lockdown_diff:.1f} µg/m³ niedriger als erwartet ({no2_lockdown_diff/during_lockdown['no2_pred'].mean()*100:.1f}%)")

# Detaillierter Plot für Lockdown-Phase
fig, ax = plt.subplots(figsize=(14, 6))
lockdown_data = test_2020_clean[(test_2020_clean.index >= "2020-03-01") & (test_2020_clean.index < "2020-06-01")]

ax.plot(lockdown_data.index, lockdown_data["pm10"], 
        label="PM10 Tatsächlich", color="blue", linewidth=2, marker='o', markersize=4)
ax.plot(lockdown_data.index, lockdown_data["pm10_pred"], 
        label="PM10 Vorhersage", color="red", linestyle="--", linewidth=2)
ax.plot(lockdown_data.index, lockdown_data["no2"], 
        label="NO2 Tatsächlich", color="cyan", linewidth=2, marker='s', markersize=4)
ax.plot(lockdown_data.index, lockdown_data["no2_pred"], 
        label="NO2 Vorhersage", color="orange", linestyle="--", linewidth=2)

ax.axvline(lockdown_start, color="green", linestyle=":", linewidth=2.5, label="Lockdown Start")
ax.axvline(lockdown_end, color="orange", linestyle=":", linewidth=2.5, label="1. Lockerungen")
ax.axvline(lockdown_end2, color="purple", linestyle=":", linewidth=2.5, label="2. Lockerungen")
ax.axvspan(lockdown_start, lockdown_end2, alpha=0.2, color="gray")

ax.set_xlabel("Datum", fontsize=12)
ax.set_ylabel("Konzentration (µg/m³)", fontsize=12)
ax.set_title("COVID-19 Lockdown Effekt: März-Mai 2020", fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/A7_covid19_lockdown_detail.png", dpi=150)
plt.close()

print("\n--- ALLE AUFGABEN ABGESCHLOSSEN ---")
print(f"Alle Plots gespeichert im Ordner 'plots/'")