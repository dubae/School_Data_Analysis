import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QScrollArea
from PyQt5.QtCore import Qt
import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우의 경우 Malgun Gothic 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit()
    
    all_data = {}
    for year in ['2019', '2020', '2021', '2022', '2023']:
        df = pd.read_excel(file_path, sheet_name=year)
        # 데이터 전처리: 2019-2022년의 "교외활동"을 "교외"로 변경
        if year in ['2019', '2020', '2021', '2022']:
            df['사고장소'] = df['사고장소'].replace('교외활동', '교외')
        all_data[year] = df
    
    return all_data

# 특정 시간대 사고 수 계산 및 장소별 비율 계산
def get_accident_counts_and_place_distribution(data, region, day, start_hour, end_hour):
    counts = {}
    place_distribution = {}
    place_counts_total = {}
    places = ['교실', '교외', '부속시설', '운동장', '통로']
    place_counts_by_year = {place: [] for place in places}

    for year, df in data.items():
        # DataFrame의 사본을 생성하여 원본을 변경하지 않음
        df_copy = df.copy()
        
        # 사고발생시각을 datetime 형식으로 변환하고 시간 추출
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        # 해당 지역, 요일 및 시간대 필터링
        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        
        # 사고 수 계산
        counts[year] = len(filtered_df)
        
        # 사고 장소별 비율 및 횟수 계산
        place_counts = filtered_df['사고장소'].value_counts()
        total = place_counts.sum()
        place_counts_total[year] = place_counts.to_dict()
        for place in places:
            place_counts_by_year[place].append(place_counts.get(place, 0))
        if total > 0:
            place_distribution[year] = {place: (place_counts.get(place, 0) / total) * 100 for place in places}
        else:
            place_distribution[year] = {place: 0 for place in places}
    
    return counts, place_distribution, place_counts_total, place_counts_by_year

# 선형 회귀를 이용한 사고 장소별 2024년 사고 수 예측
def predict_accidents_by_place(data, region, day, start_hour, end_hour):
    place_counts_by_year = {place: [] for place in ['교실', '교외', '부속시설', '운동장', '통로']}
    years = []

    for year, df in data.items():
        # DataFrame의 사본을 생성하여 원본을 변경하지 않음
        df_copy = df.copy()
        
        # 사고발생시각을 datetime 형식으로 변환하고 시간 추출
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        # 해당 지역, 요일 및 시간대 필터링
        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        
        # 사고 장소별 사고 수 계산
        place_counts = filtered_df['사고장소'].value_counts()
        for place in place_counts_by_year.keys():
            place_counts_by_year[place].append(place_counts.get(place, 0))
        
        years.append(int(year))
    
    # 선형 회귀 모델 학습 및 2024년 예측
    predicted_counts_2024 = {}
    regression_models = {}
    X = np.array(years).reshape(-1, 1)
    for place, counts in place_counts_by_year.items():
        if len(counts) == len(years):
            y = np.array(counts)
            model = LinearRegression().fit(X, y)
            predicted_counts_2024[place] = model.predict(np.array([[2024]]))[0]
            regression_models[place] = model
        else:
            predicted_counts_2024[place] = 0

    total_predicted_count = sum(predicted_counts_2024.values())
    predicted_percentage_2024 = {place: (count / total_predicted_count) * 100 if total_predicted_count > 0 else 0 for place, count in predicted_counts_2024.items()}

    return predicted_counts_2024, total_predicted_count, predicted_percentage_2024, regression_models, place_counts_by_year

# PyQt GUI
class MainWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        
        self.setWindowTitle("지역별 학교 안전사고 수")
        
        self.layout = QVBoxLayout()
        
        self.region_label = QLabel("지역:")
        self.region_input = QLineEdit()
        self.day_label = QLabel("요일 (월, 화, 수, 목, 금, 토, 일):")
        self.day_input = QLineEdit()
        self.start_hour_label = QLabel("시작 시간 (0-23):")
        self.start_hour_input = QLineEdit()
        self.end_hour_label = QLabel("종료 시간 (1-24):")
        self.end_hour_input = QLineEdit()
        self.predict_button = QPushButton("사고 수 확인")
        
        self.tab_widget = QTabWidget()
        self.predicted_table = QTableWidget()
        self.history_table = QTableWidget()
        self.chart_scroll_area = QScrollArea()
        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout()
        self.chart_widget.setLayout(self.chart_layout)
        self.chart_scroll_area.setWidgetResizable(True)
        self.chart_scroll_area.setWidget(self.chart_widget)
        
        self.tab_widget.addTab(self.predicted_table, "2024년 예측 사고 데이터")
        self.tab_widget.addTab(self.history_table, "과거 사고 데이터")
        self.tab_widget.addTab(self.chart_scroll_area, "원형 그래프 및 회귀 모델")
        
        self.layout.addWidget(self.region_label)
        self.layout.addWidget(self.region_input)
        self.layout.addWidget(self.day_label)
        self.layout.addWidget(self.day_input)
        self.layout.addWidget(self.start_hour_label)
        self.layout.addWidget(self.start_hour_input)
        self.layout.addWidget(self.end_hour_label)
        self.layout.addWidget(self.end_hour_input)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.tab_widget)
        
        self.predict_button.clicked.connect(self.show_accident_counts)
        
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
    
    def show_accident_counts(self):
        region = self.region_input.text()
        day = self.day_input.text()
        start_hour = int(self.start_hour_input.text())
        end_hour = int(self.end_hour_input.text())
        
        counts, place_distribution, place_counts_total, place_counts_by_year = get_accident_counts_and_place_distribution(self.data, region, day, start_hour, end_hour)
        predicted_counts_2024, total_predicted_count, predicted_percentage_2024, regression_models, place_counts_by_year = predict_accidents_by_place(self.data, region, day, start_hour, end_hour)
        
        # 2024년 예측 테이블 업데이트
        self.predicted_table.clear()
        self.predicted_table.setRowCount(0)
        self.predicted_table.setColumnCount(4)
        self.predicted_table.setHorizontalHeaderLabels(['Year', 'Place', 'Count', 'Percentage'])
        
        row = 0
        for place, count in predicted_counts_2024.items():
            percentage = predicted_percentage_2024[place]
            self.predicted_table.insertRow(row)
            self.predicted_table.setItem(row, 0, QTableWidgetItem('2024'))
            self.predicted_table.setItem(row, 1, QTableWidgetItem(place))
            self.predicted_table.setItem(row, 2, QTableWidgetItem(f"{count:.2f} cases"))
            self.predicted_table.setItem(row, 3, QTableWidgetItem(f"{percentage:.2f}%"))
            row += 1
        
        self.predicted_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 과거 데이터 테이블 업데이트
        self.history_table.clear()
        self.history_table.setRowCount(0)
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(['Year', 'Place', 'Count', 'Percentage'])
        
        row = 0
        for year, distribution in place_distribution.items():
            for place, percentage in distribution.items():
                count = place_counts_total[year].get(place, 0)
                self.history_table.insertRow(row)
                self.history_table.setItem(row, 0, QTableWidgetItem(str(year)))
                self.history_table.setItem(row, 1, QTableWidgetItem(place))
                self.history_table.setItem(row, 2, QTableWidgetItem(f"{count} cases"))
                self.history_table.setItem(row, 3, QTableWidgetItem(f"{percentage:.2f}%"))
                row += 1
        
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 이전에 생성된 차트를 삭제합니다.
        for i in reversed(range(self.chart_layout.count())): 
            widget = self.chart_layout.itemAt(i).widget()
            if widget is not None: 
                widget.setParent(None)

        # 년도별 원형 그래프 생성
        for year, distribution in place_distribution.items():
            labels = list(distribution.keys())
            sizes = list(distribution.values())
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title(f'{year} Accident Place Distribution')
            
            chart_widget = QWidget()
            chart_layout = QVBoxLayout()
            chart_label = QLabel()
            chart_label.setPixmap(self.fig_to_pixmap(fig))
            chart_layout.addWidget(chart_label)
            chart_widget.setLayout(chart_layout)
            self.chart_layout.addWidget(chart_widget)

        # 2024년 예측 원형 그래프 생성
        labels = list(predicted_percentage_2024.keys())
        sizes = list(predicted_percentage_2024.values())
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('2024 Accident Place Distribution Prediction')
        
        chart_widget = QWidget()
        chart_layout = QVBoxLayout()
        chart_label = QLabel()
        chart_label.setPixmap(self.fig_to_pixmap(fig))
        chart_layout.addWidget(chart_label)
        chart_widget.setLayout(chart_layout)
        self.chart_layout.addWidget(chart_widget)

        # 선형 회귀 모델 시각화
        years = np.array([int(year) for year in place_distribution.keys()]).reshape(-1, 1)
        for place, model in regression_models.items():
            fig, ax = plt.subplots()
            ax.scatter(years, place_counts_by_year[place], color='blue')
            ax.plot(years, model.predict(years), color='red')
            ax.set_title(f'{place} Accident Trend')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Accidents')
            
            chart_widget = QWidget()
            chart_layout = QVBoxLayout()
            chart_label = QLabel()
            chart_label.setPixmap(self.fig_to_pixmap(fig))
            chart_layout.addWidget(chart_label)
            chart_widget.setLayout(chart_layout)
            self.chart_layout.addWidget(chart_widget)
    
    def fig_to_pixmap(self, fig):
        import io
        from PyQt5.QtGui import QPixmap, QImage
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = QImage()
        image.loadFromData(buf.read())
        return QPixmap(image)

if __name__ == '__main__':
    file_path = 'schoolData.xlsx'
    data = load_and_preprocess_data(file_path)
    
    app = QApplication(sys.argv)
    window = MainWindow(data)
    window.show()
    sys.exit(app.exec_())
