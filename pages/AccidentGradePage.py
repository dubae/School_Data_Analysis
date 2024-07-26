import pandas as pd
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QTextEdit
from sklearn.linear_model import LinearRegression
import numpy as np

def load_and_preprocess_data(file_path):
    all_data = {}
    for year in ['2019', '2020', '2021', '2022', '2023']:
        df = pd.read_excel(file_path, sheet_name=year)
        # 데이터 전처리: 2019-2022년의 "교외활동"을 "교외"로 변경
        if year in ['2019', '2020', '2021', '2022']:
            df['사고장소'] = df['사고장소'].regrade('교외활동', '교외')
        all_data[year] = df
    
    return all_data

# 특정 시간대 사고 수 계산 및 장소별 비율 계산
def get_accident_counts_and_grade_distribution(data, region, day, start_hour, end_hour):
    counts = {}
    grade_distribution = {}
    grade_counts_total = {}
    grades = ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년', '유아', 'N/A']

    for year, df in data.items():
        df_copy = df.copy()
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        counts[year] = len(filtered_df)
        
        grade_counts = filtered_df['사고자학년'].value_counts()
        total = grade_counts.sum()
        grade_counts_total[year] = grade_counts.to_dict()
        if total > 0:
            grade_distribution[year] = {grade: (grade_counts.get(grade, 0) / total) * 100 for grade in grades}
        else:
            grade_distribution[year] = {grade: 0 for grade in grades}
    
    return counts, grade_distribution, grade_counts_total

# 선형 회귀를 이용한 사고 장소별 2024년 사고 수 예측
def predict_accidents_by_grade(data, region, day, start_hour, end_hour):
    grade_counts_by_year = {grade: [] for grade in ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년', '유아', 'N/A']}
    years = []

    for year, df in data.items():
        df_copy = df.copy()
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        grade_counts = filtered_df['사고자학년'].value_counts()
        for grade in grade_counts_by_year.keys():
            grade_counts_by_year[grade].append(grade_counts.get(grade, 0))
        
        years.append(int(year))
    
    predicted_counts_2024 = {}
    X = np.array(years).reshape(-1, 1)
    for grade, counts in grade_counts_by_year.items():
        if len(counts) == len(years):
            y = np.array(counts)
            model = LinearRegression().fit(X, y)
            predicted_counts_2024[grade] = model.predict(np.array([[2024]]))[0]
        else:
            predicted_counts_2024[grade] = 0

    total_predicted_count = sum(predicted_counts_2024.values())
    predicted_percentage_2024 = {grade: (count / total_predicted_count) * 100 if total_predicted_count > 0 else 0 for grade, count in predicted_counts_2024.items()}

    return predicted_counts_2024, total_predicted_count, predicted_percentage_2024

class AccidentGradePage(QWidget):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        
        self.setWindowTitle("사고장소 페이지")
        
        self.layout = QVBoxLayout()
        self.title = QLabel("사고학년 페이지")

        self.region_label = QLabel("지역:")
        self.region_input = QLineEdit()
        self.day_label = QLabel("요일 (월, 화, 수, 목, 금, 토, 일):")
        self.day_input = QLineEdit()
        self.start_hour_label = QLabel("시작 시간 (0-23):")
        self.start_hour_input = QLineEdit()
        self.end_hour_label = QLabel("종료 시간 (1-24):")
        self.end_hour_input = QLineEdit()
        self.predict_button = QPushButton("사고 수 확인")
        self.result_label = QTextEdit()
        
        self.layout.addWidget(self.region_label)
        self.layout.addWidget(self.region_input)
        self.layout.addWidget(self.day_label)
        self.layout.addWidget(self.day_input)
        self.layout.addWidget(self.start_hour_label)
        self.layout.addWidget(self.start_hour_input)
        self.layout.addWidget(self.end_hour_label)
        self.layout.addWidget(self.end_hour_input)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.result_label)
        
        self.predict_button.clicked.connect(self.show_accident_counts)
        
        self.setLayout(self.layout)
    
    def show_accident_counts(self):
        region = self.region_input.text()
        day = self.day_input.text()
        start_hour = int(self.start_hour_input.text())
        end_hour = int(self.end_hour_input.text())
        
        counts, grade_distribution, grade_counts_total = get_accident_counts_and_grade_distribution(self.data, region, day, start_hour, end_hour)
        predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_grade(self.data, region, day, start_hour, end_hour)
        
        result_text = f"{region} 지역에서 {day}요일 {start_hour}시부터 {end_hour}시까지의 사고 수:\n"
        for year, count in counts.items():
            result_text += f"{year}: {count}건\n"
        
        result_text += "\n사고 학년별 비율 및 횟수:\n"
        for year, distribution in grade_distribution.items():
            result_text += f"{year}년:\n"
            for grade, percentage in distribution.items():
                count = grade_counts_total[year].get(grade, 0)
                result_text += f"  {grade}: {count}건 ({percentage:.2f}%)\n"
        
        result_text += f"\n2024년 {day}요일에 예측된 사고 수:\n"
        for grade, count in predicted_counts_2024.items():
            percentage = predicted_percentage_2024[grade]
            result_text += f"  {grade}: {count:.2f}건 ({percentage:.2f}%)\n"
        result_text += f"\n총 예측 사고 수: {total_predicted_count:.2f}건\n"
        
        # 두회의 테스트
        result_text += f"\n--------------TEST--------------\n"
        for i in range(start_hour,24,1):
            predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_grade(self.data, region, day, i, i+1)
            result_text += f"\n2024년 {day}요일 {i}~{i+1}에 예측된 사고 수:\n"
            for grade, count in predicted_counts_2024.items():
                percentage = predicted_percentage_2024[grade]
                result_text += f"  {grade}: {count:.2f}건 ({percentage:.2f}%)\n"
            result_text += f"\n총 예측 사고 수: {total_predicted_count:.2f}건\n"
        
        self.result_label.setText(result_text)
