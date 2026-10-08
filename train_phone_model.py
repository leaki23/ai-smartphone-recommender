import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading data...")
df = pd.read_csv('dataset/phone_data_1000.csv')
print(f"✅ Loaded {len(df)} rows")

X = df[['Budget', 'Storage_GB', 'OS_Choice', 'Camera_Importance', 'Battery_Importance', 'Gaming_Importance']]
y = df['Recommended_Model'] 

print(f"Target classes: {y.unique()}")

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


with open('phone_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'phone_model.pkl'!")
print(f"Accuracy: {model.score(X, y) * 100:.2f}%")