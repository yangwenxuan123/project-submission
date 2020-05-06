import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import tkinter as tk
# Input data files are available in the "input/" directory.

df = pd.read_csv('input/2016-17_teamBoxScore.csv')
feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']

# PTS - Points
# DRTG - Defensive Rating
# PF - Personal Fouls
# TO - Turnover
# ORB - Offensive Rebounds
# FGA - Field Goal Attempts

x = df[feature_cols]
y = df['teamRslt']

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.4, random_state=2)

clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)


def show_prediction():
    arr = [[ e1.get(), (float(e2.get())*4), e3.get(), e4.get(), e5.get(), e6.get()]]
    prediction = clfgtb.predict(arr)
    entry_I2.configure(text=str(prediction[0]))

master = tk.Tk()
master.title("NBA Predictor")
master.geometry("300x300")
tk.Label(master, text="NBA Win/Loss Predictor").grid(row=0)
tk.Label(master, text="Opponent Points").grid(row=2)
tk.Label(master, text="Team Defensive Rating").grid(row=3)
tk.Label(master, text="Team Personal Fouls").grid(row=4)
tk.Label(master, text="Team Turnover").grid(row=5)
tk.Label(master, text="Team Offensive Rebounds").grid(row=6)
tk.Label(master, text="Team Field Goal Attempts").grid(row=7)

e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)
e4 = tk.Entry(master)
e5 = tk.Entry(master)
e6 = tk.Entry(master)

e1.grid(row=2, column=1)
e2.grid(row=3, column=1)
e3.grid(row=4, column=1)
e4.grid(row=5, column=1)
e5.grid(row=6, column=1)
e6.grid(row=7, column=1)

tk.Button(master, text='Quit', command=master.quit).grid(row=8, column=0, sticky=tk.W, pady=4)
tk.Button(master, text='Show', command=show_prediction).grid(row=8, column=1, sticky=tk.W, pady=4)

label_I2 = tk.Label(master, text="Prediction")
label_I2.grid(row=9, column=0)

entry_I2 = tk.Label(master, width=15, height=1, bg="light grey")
entry_I2.grid(row=9, column=1)

tk.mainloop()