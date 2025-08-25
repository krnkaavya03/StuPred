import pandas as pd
import random

data = []

for i in range(1, 501):  # 500 students
    student_id = f"S{i:03d}"
    attendance = random.randint(40, 100)
    study_hours = random.randint(0, 70)
    assignments_done = random.randint(0, 10)
    previous_grade = random.randint(35, 100)
    midterm_score = random.randint(35, 100)
    participations = random.randint(0, 10)
    active_extracurricular = random.randint(0, 5)
    
    # Adjusted success rule for realistic balance
    # Weighted sum
    score = (attendance*0.25 + previous_grade*0.25 + study_hours*1.5 + 
             assignments_done*1.2 + midterm_score*0.2 + participations*2 + active_extracurricular*1.5)
    
    # Lower threshold to get balanced success distribution
    success = 1 if score >= 100 else 0
    
    data.append([student_id, attendance, study_hours, assignments_done, previous_grade,
                 midterm_score, participations, active_extracurricular, success])

df = pd.DataFrame(data, columns=[
    'student_id', 'attendance','study_hours','assignments_done','previous_grade',
    'midterm_score','participations','active_extracurricular','success'
])

df.to_csv('students.csv', index=False)
print("Balanced dataset created: students.csv")
print("Success distribution:\n", df['success'].value_counts())


