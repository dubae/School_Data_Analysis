import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QLabel, QComboBox, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QBrush, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

def load_and_preprocess_data(file_path):
    all_data = {}
    for year in ['2019', '2020', '2021', '2022', '2023']:
        df = pd.read_excel(file_path, sheet_name=year)
        # 데이터 전처리: 2019-2022년의 "교외활동"을 "교외"로 변경
        if year in ['2019', '2020', '2021', '2022']:
            df['사고장소'] = df['사고장소'].replace('교외활동', '교외')
            df.rename(columns={'사고매개물': '매개물'}, inplace=True)
        all_data[year] = df
    
    return all_data

# 특정 시간대 사고 수 계산 및 장소별 비율 계산
def get_accident_counts_and_place_distribution(data, region, day, start_hour, end_hour):
    counts = {}
    place_distribution = {}
    place_counts_total = {}
    places = ['교실', '교외', '부속시설', '운동장', '통로']

    for year, df in data.items():
        df_copy = df.copy()
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        counts[year] = len(filtered_df)
        
        place_counts = filtered_df['사고장소'].value_counts()
        total = place_counts.sum()
        place_counts_total[year] = place_counts.to_dict()
        if total > 0:
            place_distribution[year] = {place: (place_counts.get(place, 0) / total) * 100 for place in places}
        else:
            place_distribution[year] = {place: 0 for place in places}
    
    return counts, place_distribution, place_counts_total

# 선형 회귀를 이용한 사고 장소별 2024년 사고 수 예측
def predict_accidents_by_place(data, region, day, start_hour, end_hour):
    place_counts_by_year = {place: [] for place in ['교실', '교외', '부속시설', '운동장', '통로']}
    years = []

    for year, df in data.items():
        df_copy = df.copy()
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        place_counts = filtered_df['사고장소'].value_counts()
        for place in place_counts_by_year.keys():
            place_counts_by_year[place].append(place_counts.get(place, 0))
        
        years.append(int(year))
    
    predicted_counts_2024 = {}
    X = np.array(years).reshape(-1, 1)
    for place, counts in place_counts_by_year.items():
        if len(counts) == len(years):
            y = np.array(counts)
            model = LinearRegression().fit(X, y)
            predicted_counts_2024[place] = model.predict(np.array([[2024]]))[0]
            if predicted_counts_2024[place]<0:
                predicted_counts_2024[place]=0
        else:
            predicted_counts_2024[place] = 0

    total_predicted_count = sum(predicted_counts_2024.values())
    predicted_percentage_2024 = {place: (count / total_predicted_count) * 100 if total_predicted_count > 0 else 0 for place, count in predicted_counts_2024.items()}

    return predicted_counts_2024, total_predicted_count, predicted_percentage_2024

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot(self, times, type_distribution):
        self.axes.clear()
        for type, data in type_distribution.items():
            self.axes.plot(times, data, label=type)
        self.axes.set_xlabel('시간')
        self.axes.set_ylabel('비율 (%)')
        self.axes.legend()
        self.draw()

class AccidentLocationPage(QWidget):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 지역 및 요일 선택
        selection_layout = QHBoxLayout()
        
        self.region_label = QLabel("지역:")
        self.region_combo = QComboBox()
        self.region_combo.addItems(["서울", "경기", "강원", "세종", "부산", "제주", "경북", "경남", "충북", "충남", "대구", "대전", "광주", "울산", "인천", "전북", "전남"])
        
        self.day_label = QLabel("요일:")
        self.day_combo = QComboBox()
        self.day_combo.addItems(["월", "화", "수", "목", "금", "토", "일"])
        
        selection_layout.addWidget(self.region_label)
        selection_layout.addWidget(self.region_combo)
        selection_layout.addWidget(self.day_label)
        selection_layout.addWidget(self.day_combo)

        

        # 버튼 및 결과
        self.view_combo = QComboBox()
        self.view_combo.addItems(["표", "꺾은선 그래프"])
        
        self.predict_button = QPushButton("사고 수 확인")
        self.result_table = QTableWidget()
        self.result_graph = PlotCanvas(self)
        layout.addLayout(selection_layout)
        layout.addWidget(self.view_combo)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_table)
        layout.addWidget(self.result_graph)
        
        self.setLayout(layout)

        # 버튼 클릭 연결
        self.predict_button.clicked.connect(self.show_accident_counts)
        self.view_combo.currentIndexChanged.connect(self.toggle_view)

        # 기본적으로 표를 보여줍니다
        self.show_accident_counts()

    def toggle_view(self):
        view_mode = self.view_combo.currentText()
        if view_mode == "표":
            self.result_table.setVisible(True)
            self.result_graph.setVisible(False)
        elif view_mode == "꺾은선 그래프":
            self.result_table.setVisible(False)
            self.result_graph.setVisible(True)

    def show_accident_counts(self):
        region = self.region_combo.currentText()
        day = self.day_combo.currentText()
        try:
            start_hour = int(datetime.now().hour)
            start_hour = 6
        except ValueError:
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            return
        
        # 시간 범위를 생성합니다.
        hours = list(range(start_hour, 22))
        places = ['교실', '교외', '부속시설', '운동장', '통로']

        # 테이블 초기화
        self.result_table.setRowCount(len(places))
        self.result_table.setColumnCount(len(hours))
        self.result_table.setHorizontalHeaderLabels([f"{hour}~{hour+1}" for hour in hours])
        self.result_table.setVerticalHeaderLabels(places)

       # 그래프 데이터 준비
        plot_times = hours
        plot_data = {place: [] for place in places}

        # 예측된 사고 수를 계산하여 테이블과 그래프에 입력합니다.
        for hour in hours:
            predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_place(self.data, region, day, hour, hour+1)
            for i, place in enumerate(places):
                percentage = predicted_percentage_2024[place]
                item = QTableWidgetItem(f"{percentage:.2f}%")
                if percentage > 30:
                    item.setBackground(QBrush(QColor(255, 0, 0,100)))  # Red background for >30%
                elif percentage < 10:
                    item.setBackground(QBrush(QColor(0, 255, 0,90)))
                self.result_table.setItem(i, hour - start_hour, item)
                plot_data[place].append(percentage)
        
        # 테이블 보기 설정
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 꺾은선 그래프 업데이트
        self.result_graph.plot(plot_times, plot_data)
        
        self.toggle_view()

