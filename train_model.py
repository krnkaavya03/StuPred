import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('students.csv')

# Features & target
X = df[['attendance','study_hours','assignments_done','previous_grade',
        'midterm_score','participations','active_extracurricular']]
y = df['success']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("student_model.pkl", "wb"))
print("Model trained and saved as student_model.pkl")

# Check accuracy
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
