import pandas as pd

data = {
    "hours": [1, 2, 3, 4, 5, 2, 4, 6],
    "scores": [35, 42, 58, 70, 85, 48, 73, 91]
}
df = pd.DataFrame(data)

# TODO 1: Display the dataset
# TODO 2: Calculate and print the average score
# TODO 3: Find and print the highest score
# TODO 4: Add a Boolean 'passed' column (score >= 50)

print(df)
score_average = df["scores"].mean()
print(score_average)
print(df["scores"].max())

if df["scores"] >= 50:
    df["Passed"] = df["scores"].bool()
    print(df)
