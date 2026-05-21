import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

age = np.random.randint(18, 65, n)

# expérience cohérente (pas plus que âge - 18)
experience = np.array([max(0, a - np.random.randint(18, 25)) for a in age])

# salaire de base
salary = (
    3000
    + experience * 800
    + age * 20
    + np.random.normal(0, 1500, n)
)

data = pd.DataFrame({
    "Age": age,
    "Experience": experience,
    "Salary": salary.astype(int)
})

data.to_csv("better_data.csv", index=False)

print(data.head())