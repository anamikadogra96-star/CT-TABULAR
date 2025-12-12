import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
import os
import datetime

class StudyPlannerModel:
    def __init__(self):
        self.rf_reg = None
        self.rf_clf = None
        self.X_columns = None
        self.load_and_train_model()
    
    def load_and_train_model(self):
        """Load dataset and train models"""
        df = pd.read_csv('StudentsPerformance.csv')
        
        # Data preprocessing
        df['gender'] = df['gender'].map({'female': 0, 'male': 1})
        df['race/ethnicity'] = df['race/ethnicity'].map({
            'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4
        })
        df['parental level of education'] = df['parental level of education'].map({
            "some high school": 0, "high school": 1, "some college": 2,
            "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
        })
        df['lunch'] = df['lunch'].map({'standard': 0, 'free/reduced': 1})
        df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})
        
        df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
        df['performance'] = pd.cut(df['average_score'], bins=[0, 60, 80, 100], labels=['Low', 'Medium', 'High'])
        
        X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
        y_reg = df['average_score']
        y_clf = df['performance']
        self.X_columns = X.columns
        
        # Train-test split
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_clf, test_size=0.2, random_state=42)

        # Store for metrics testing
        self.X_test = X_test
        self.y_test_reg = y_test_reg
        self.X_test_cls = X_test_cls
        self.y_test_cls = y_test_cls

        # Train models
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        self.rf_reg.fit(X_train, y_train_reg)
        self.rf_clf.fit(X_train_cls, y_train_cls)

        # ---- 📌 PERFORMANCE METRICS PRINT HERE ----
        self.print_metrics()
        # -------------------------------------------

    def print_metrics(self):
        """Print evaluation metrics in terminal"""
        y_pred_reg = self.rf_reg.predict(self.X_test)
        y_pred_cls = self.rf_clf.predict(self.X_test_cls)

        rmse = np.sqrt(mean_squared_error(self.y_test_reg, y_pred_reg))
        r2 = r2_score(self.y_test_reg, y_pred_reg)
        precision = precision_score(self.y_test_cls, y_pred_cls, average='macro')
        recall = recall_score(self.y_test_cls, y_pred_cls, average='macro')
        f1 = f1_score(self.y_test_cls, y_pred_cls, average='macro')

        print("\n============================")
        print("📌 MODEL PERFORMANCE METRICS")
        print("============================")
        print(f"RMSE:                   {rmse:.3f}")
        print(f"R² Score:               {r2:.3f}")
        print(f"Precision:              {precision:.3f}")
        print(f"Recall:                 {recall:.3f}")
        print(f"F1 Score:               {f1:.3f}")
        print("============================\n")

    def predict(self, gender, race, parent_edu, lunch, test_prep):
        input_data = pd.DataFrame([[gender, race, parent_edu, lunch, test_prep]], 
                                 columns=self.X_columns)
        score = self.rf_reg.predict(input_data)[0]
        level = self.rf_clf.predict(input_data)[0]
        return score, level
    
    def generate_timetable(self, score):
        if score < 60:
            math_h, reading_h, writing_h = 3.0, 2.0, 2.0
            total_h = 7.0
        elif score < 80:
            math_h, reading_h, writing_h = 2.0, 1.5, 1.5
            total_h = 5.0
        else:
            math_h, reading_h, writing_h = 1.5, 1.0, 1.0
            total_h = 3.5
        
        timetable = [
            {"time": "06:00 AM", "activity": "Wake Up + Morning Exercise (30 min)"},
            {"time": "06:30 AM", "activity": "Breakfast"},
            {"time": "07:00 AM", "activity": f"MATH STUDY ({math_h} hours)"},
            {"time": "09:00 AM", "activity": "BREAK (10 min)"},
            {"time": "09:10 AM", "activity": f"READING ({reading_h} hours)"},
            {"time": "11:00 AM", "activity": "LUNCH + REST"},
            {"time": "01:30 PM", "activity": f"WRITING ({writing_h} hours)"},
            {"time": "03:30 PM", "activity": "POMODORO BREAK (15 min)"},
            {"time": "03:45 PM", "activity": "REVISION + WEAK TOPICS (1 hour)"},
            {"time": "04:45 PM", "activity": "OUTDOOR ACTIVITY"},
            {"time": "06:00 PM", "activity": "MOCK TEST / PAST PAPERS (1 hour)"},
            {"time": "08:00 PM", "activity": "DINNER"},
            {"time": "09:00 PM", "activity": "LIGHT REVISION (30 min)"},
            {"time": "09:30 PM", "activity": "SLEEP (8 hours recommended)"}
        ]
        
        return {
            'timetable': timetable,
            'math_hours': math_h,
            'reading_hours': reading_h,
            'writing_hours': writing_h,
            'total_hours': total_h
        }
    
    def save_progress(self, math_h, reading_h, writing_h):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        total = math_h + reading_h + writing_h
        target = 5.0
        progress = min(100, (total / target) * 100)
        
        new_entry = pd.DataFrame([{
            "Date": today,
            "Math_Hours": math_h,
            "Reading_Hours": reading_h,
            "Writing_Hours": writing_h,
            "Total_Hours": total,
            "Progress_%": progress
        }])
        
        if os.path.exists("progress_log.csv"):
            log_df = pd.read_csv("progress_log.csv")
            log_df = pd.concat([log_df, new_entry], ignore_index=True)
        else:
            log_df = new_entry
        
        log_df.to_csv("progress_log.csv", index=False)
        return progress
    
    def get_progress_data(self):
        if os.path.exists("progress_log.csv"):
            df = pd.read_csv("progress_log.csv")
            return df.to_dict('records')
        return []
