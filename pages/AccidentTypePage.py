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
            df['사고장소'] = df['사고장소'].retype('교외활동', '교외')
        all_data[year] = df
    
    return all_data

# 특정 시간대 사고 수 계산 및 장소별 비율 계산
def get_accident_counts_and_type_distribution(data, region, day, start_hour, end_hour):
    counts = {}
    type_distribution = {}
    type_counts_total = {}
    types = ['기타', '낙상', '낙상-넘어짐', '낙상-떨어짐', '낙상-미끄러짐', '물리적힘 노출', '사람과의 충돌', '염좌·삐임 등 신체 충격']

    for year, df in data.items():
        df_copy = df.copy()
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        counts[year] = len(filtered_df)
        
        type_counts = filtered_df['사고형태'].value_counts()
        total = type_counts.sum()
        type_counts_total[year] = type_counts.to_dict()
        if total > 0:
            type_distribution[year] = {type: (type_counts.get(type, 0) / total) * 100 for type in types}
        else:
            type_distribution[year] = {type: 0 for type in types}
    
    return counts, type_distribution, type_counts_total

# 선형 회귀를 이용한 사고 장소별 2024년 사고 수 예측
def predict_accidents_by_type(data, region, day, start_hour, end_hour):
    type_counts_by_year = {type: [] for type in ['기타', '낙상', '낙상-넘어짐', '낙상-떨어짐', '낙상-미끄러짐', '물리적힘 노출', '사람과의 충돌', '염좌·삐임 등 신체 충격']}
    years = []

    for year, df in data.items():
        df_copy = df.copy()
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        type_counts = filtered_df['사고형태'].value_counts()
        for type in type_counts_by_year.keys():
            type_counts_by_year[type].append(type_counts.get(type, 0))
        
        years.append(int(year))
    
    predicted_counts_2024 = {}
    X = np.array(years).reshape(-1, 1)
    for type, counts in type_counts_by_year.items():
        if len(counts) == len(years):
            y = np.array(counts)
            model = LinearRegression().fit(X, y)
            predicted_counts_2024[type] = model.predict(np.array([[2024]]))[0]
            if predicted_counts_2024[type]<0:
                predicted_counts_2024[type]=0
        else:
            predicted_counts_2024[type] = 0

    total_predicted_count = sum(predicted_counts_2024.values())
    predicted_percentage_2024 = {type: (count / total_predicted_count) * 100 if total_predicted_count > 0 else 0 for type, count in predicted_counts_2024.items()}

    return predicted_counts_2024, total_predicted_count, predicted_percentage_2024

class AccidentTypePage(QWidget):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        
        self.setWindowTitle("사고부위 페이지")
        
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
        
        # # 시간 입력
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
            start_hour = int(datetime.now().hour)
        except ValueError:
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            return
        
        # 시간 범위를 생성합니다.
        hours = list(range(start_hour, 24))
        types = ['기타', '낙상', '낙상-넘어짐', '낙상-떨어짐', '낙상-미끄러짐', '물리적힘 노출', '사람과의 충돌', '염좌·삐임 등 신체 충격']

        # 테이블 초기화
        self.result_table.setRowCount(len(types))
        self.result_table.setColumnCount(len(hours))
        self.result_table.setHorizontalHeaderLabels([f"{hour}~{hour+1}" for hour in hours])
        self.result_table.setVerticalHeaderLabels(types)

        # 예측된 사고 수를 계산하여 테이블에 입력합니다.
        for i, hour in enumerate(hours):
            predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_type(self.data, region, day, hour, hour+1)
            for j, type in enumerate(types):
                percentage = predicted_percentage_2024[type]
                item = QTableWidgetItem(f"{percentage:.2f}%")
                if percentage > 30:
                    item.setBackground(QBrush(QColor(255, 0, 0,100)))  # Red background for >30%
                self.result_table.setItem(j, i, item)

        # 테이블 보기 설정
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)


