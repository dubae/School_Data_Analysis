import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, 
                             QComboBox, QStackedWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.linear_model import LinearRegression
import numpy as np
import io

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우의 경우 Malgun Gothic 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

class AccidentGraphPage(QWidget):
    def __init__(self, data):
        super().__init__()
        self.data = data

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.year_selector = QComboBox()
        self.year_selector.addItems(["2019", "2020", "2021", "2022", "2023", "2024"])
        self.year_selector.currentIndexChanged.connect(self.update_charts)
        self.layout.addWidget(self.year_selector)

        self.pie_chart_label = QLabel()
        self.layout.addWidget(self.pie_chart_label)

        self.regression_chart_label = QLabel()
        self.layout.addWidget(self.regression_chart_label)

        self.create_charts("2019")

    def create_charts(self, year):
        # 데이터를 준비합니다.
        counts, place_distribution, _ = get_accident_counts_and_place_distribution(
            self.data, "서울", 0, 24)

        # 2024년 예측 결과 준비
        predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_place(
            self.data, "서울", "월", 0, 24)

        # 원형 차트를 생성합니다.
        if year == "2024":
            labels = list(predicted_percentage_2024.keys())
            sizes = list(predicted_percentage_2024.values())
        else:
            year = int(year)
            if year not in place_distribution:
                self.pie_chart_label.setText(f"{year}년도 데이터가 없습니다.")
                return
            labels = list(place_distribution[year].keys())
            sizes = list(place_distribution[year].values())

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        self.pie_chart_label.setPixmap(self.fig_to_pixmap(fig))

        # 회귀 차트를 생성합니다.
        years = list(range(2019, 2024))
        regression_data = []
        for year in years:
            if year in place_distribution:
                regression_data.append(sum(place_distribution[year].values()))
            else:
                regression_data.append(0)

        X = np.array(years).reshape(-1, 1)
        y = np.array(regression_data)
        model = LinearRegression().fit(X, y)
        future_years = np.array([[2024]])
        future_pred = model.predict(future_years)

        fig, ax = plt.subplots()
        ax.plot(years, y, label="Actual")
        ax.plot(future_years, future_pred, label="Predicted", linestyle='--')
        ax.set_xlabel('Year')
        ax.set_ylabel('Accidents')
        ax.set_title('Accident Prediction')
        ax.legend()
        self.regression_chart_label.setPixmap(self.fig_to_pixmap(fig))

    def update_charts(self):
        selected_year = self.year_selector.currentText()
        self.create_charts(selected_year)

    def fig_to_pixmap(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = QImage()
        image.loadFromData(buf.read())
        return QPixmap(image)

def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit()
    
    all_data = {}
    for year in ['2019', '2020', '2021', '2022', '2023']:
        df = pd.read_excel(file_path, sheet_name=year)
        if year in ['2019', '2020', '2021', '2022']:
            df['사고장소'] = df['사고장소'].replace('교외활동', '교외')
        all_data[year] = df
    
    return all_data

def get_accident_counts_and_place_distribution(data, region, start_hour, end_hour):
    counts = {}
    place_distribution = {}
    places = ['교실', '교외', '부속시설', '운동장', '통로']

    for year, df in data.items():
        df_copy = df.copy()
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        
        counts[year] = len(filtered_df)
        
        place_counts = filtered_df['사고장소'].value_counts()
        total = place_counts.sum()
        if total > 0:
            place_distribution[year] = {place: (place_counts.get(place, 0) / total) * 100 for place in places}
        else:
            place_distribution[year] = {place: 0 for place in places}
    
    return counts, place_distribution, None

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
        else:
            predicted_counts_2024[place] = 0

    total_predicted_count = sum(predicted_counts_2024.values())
    predicted_percentage_2024 = {place: (count / total_predicted_count) * 100 if total_predicted_count > 0 else 0 for place, count in predicted_counts_2024.items()}

    return predicted_counts_2024, total_predicted_count, predicted_percentage_2024
