import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("student_exam_scores.csv")
Y = df["exam_score"]
features = df.columns[1:5]


for feature in features:
    X = df[[feature]]

    lr = LinearRegression()
    lr.fit(X, Y)
    yhat = lr.predict(X)

    plt.scatter(X, Y, label="Actual Data")
    plt.plot(X, yhat, color="red", label="Regression Line")
    plt.xlabel(feature)
    plt.ylabel("Exam Score")
    plt.title(f"Linear Regression of Exam Scores and {feature}")
    plt.legend()
    plt.show()