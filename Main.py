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
