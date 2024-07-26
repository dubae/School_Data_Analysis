import pandas as pd
from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtGui import QBrush, QColor
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime


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
            if predicted_counts_2024[grade]<0:
                predicted_counts_2024[grade]=0
        else:
            predicted_counts_2024[grade] = 0

    total_predicted_count = sum(predicted_counts_2024.values())
    predicted_percentage_2024 = {grade: (count / total_predicted_count) * 100 if total_predicted_count > 0 else 0 for grade, count in predicted_counts_2024.items()}

    return predicted_counts_2024, total_predicted_count, predicted_percentage_2024

class AccidentGradePage(QWidget):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        
        self.setWindowTitle("사고학년 페이지")
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 지역 및 요일 선택
        selection_layout = QHBoxLayout()
        
        self.region_label = QLabel("지역:")
        self.region_combo = QComboBox()
        self.region_combo.addItems(["서울", "경기", "강원", "세종", "부산", "제주"])
        
        self.day_label = QLabel("요일:")
        self.day_combo = QComboBox()
        self.day_combo.addItems(["월", "화", "수", "목", "금", "토", "일"])
        
        selection_layout.addWidget(self.region_label)
        selection_layout.addWidget(self.region_combo)
        selection_layout.addWidget(self.day_label)
        selection_layout.addWidget(self.day_combo)
        
        # 시간 입력
        # self.start_hour_label = QLabel("시작 시간 (0-23):")
        # self.start_hour_input = QLineEdit()
        # self.end_hour_label = QLabel("종료 시간 (1-24):")
        # self.end_hour_input = QLineEdit()

        # 버튼 및 결과
        self.predict_button = QPushButton("사고 수 확인")
        self.result_table = QTableWidget()
        
        # 레이아웃에 위젯 추가
        layout.addLayout(selection_layout)
        # layout.addWidget(self.start_hour_label)
        # layout.addWidget(self.start_hour_input)
        # layout.addWidget(self.end_hour_label)
        # layout.addWidget(self.end_hour_input)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_table)
        
        self.setLayout(layout)

        # 버튼 클릭 연결
        self.show_accident_counts()
        self.predict_button.clicked.connect(self.show_accident_counts)
    
    def show_accident_counts(self):
        region = self.region_combo.currentText()
        day = self.day_combo.currentText()
        try:
            # start_hour = int(self.start_hour_input.text())
            # end_hour = int(self.end_hour_input.text())
            start_hour=int(datetime.now().hour)
        except ValueError:
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            return
        
        # 시간 범위를 생성합니다.
        hours = list(range(start_hour, 24))
        grades = ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년', '유아', 'N/A']

        # 테이블 초기화
        self.result_table.setRowCount(len(grades))
        self.result_table.setColumnCount(len(hours))
        self.result_table.setHorizontalHeaderLabels([f"{hour}~{hour+1}" for hour in hours])
        self.result_table.setVerticalHeaderLabels(grades)

        # 예측된 사고 수를 계산하여 테이블에 입력합니다.
        for i, hour in enumerate(hours):
            predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_grade(self.data, region, day, hour, hour+1)
            for j, grade in enumerate(grades):
                percentage = predicted_percentage_2024[grade]
                item = QTableWidgetItem(f"{percentage:.2f}%")
                if percentage > 30:
                    item.setBackground(QBrush(QColor(255, 0, 0,100)))  # Red background for >30%
                self.result_table.setItem(j, i, item)

        # 테이블 보기 설정
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)