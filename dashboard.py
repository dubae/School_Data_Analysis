import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    all_data = {}
    for year in ['2019', '2020', '2021', '2022', '2023']:
        df = pd.read_excel(file_path, sheet_name=year)
        # 데이터 전처리: 2019-2022년의 "교외활동"을 "교외"로 변경
        if year in ['2019', '2020', '2021', '2022']:
            df['사고장소'] = df['사고장소'].replace('교외활동', '교외')
        all_data[year] = df
    return all_data

# 특정 시간대 사고 수 계산 및 장소별 비율 계산
def get_hourly_accident_counts_and_place_distribution(data, region, start_hour, day):
    counts = {}
    place_distribution = {}
    places = ['교실', '교외', '부속시설', '운동장', '통로']

    for year, df in data.items():
        # DataFrame의 사본을 생성하여 원본을 변경하지 않음
        df_copy = df.copy()
        
        # 사고발생시각을 datetime 형식으로 변환하고 시간 추출
        try:
            df_copy['사고발생시각'] = pd.to_datetime(df_copy['사고발생시각'], format='%H:%M', errors='coerce').dt.hour
        except Exception as e:
            print(f"Error processing 사고발생시각 in year {year}: {e}")
            continue

        # 요일 필터링
        df_copy = df_copy[df_copy['사고발생요일'] == day]
        
        counts[year] = {}
        place_distribution[year] = {}

        for hour in range(start_hour, 24):
            # 해당 지역 및 시간대 필터링
            filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생시각'] == hour)]
            
            # 사고 수 계산
            counts[year][hour] = len(filtered_df)
            
            # 사고 장소별 비율 계산
            place_counts = filtered_df['사고장소'].value_counts()
            total = place_counts.sum()
            if total > 0:
                place_distribution[year][hour] = {place: (place_counts.get(place, 0) / total) * 100 for place in places}
            else:
                place_distribution[year][hour] = {place: 0 for place in places}
    
    return counts, place_distribution

# 2024년 사고 수 예측
def predict_2024_accidents(counts):
    hours = list(range(24))
    years = ['2019', '2020', '2021', '2022', '2023']
    predictions = {}

    for hour in hours:
        X = np.array([[int(year), hour] for year in years])
        y = np.array([counts[year].get(hour, 0) for year in years])

        model = LinearRegression()
        model.fit(X, y)
        
        pred_2024 = model.predict([[2024, hour]])
        predictions[hour] = max(0, int(pred_2024[0]))

    return predictions

# PyQt GUI
class MainWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        
        self.setWindowTitle("지역별 학교 안전사고 수")
        
        self.layout = QVBoxLayout()
        
        self.region_label = QLabel("지역:")
        self.region_input = QLineEdit()
        self.start_hour_label = QLabel("시작 시간 (0-23):")
        self.start_hour_input = QLineEdit()
        self.day_label = QLabel("요일 (월, 화, 수, 목, 금, 토, 일):")
        self.day_input = QLineEdit()
        self.predict_button = QPushButton("사고 수 확인")
        self.result_table = QTableWidget()
        
        self.layout.addWidget(self.region_label)
        self.layout.addWidget(self.region_input)
        self.layout.addWidget(self.start_hour_label)
        self.layout.addWidget(self.start_hour_input)
        self.layout.addWidget(self.day_label)
        self.layout.addWidget(self.day_input)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.result_table)
        
        self.predict_button.clicked.connect(self.show_accident_counts)
        
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
    
    def show_accident_counts(self):
        region = self.region_input.text()
        start_hour = int(self.start_hour_input.text())
        day = self.day_input.text()
        
        counts, place_distribution = get_hourly_accident_counts_and_place_distribution(self.data, region, start_hour, day)
        predictions_2024 = predict_2024_accidents(counts)
        
        # 결과 표 설정
        self.result_table.clear()
        
        # 연도별 사고 수 및 장소별 비율 설정
        total_rows = (24 - start_hour) * (2 + len(['교실', '교외', '부속시설', '운동장', '통로']))
        self.result_table.setRowCount(total_rows)
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels(['구분', '2019', '2020', '2021', '2022', '2023', '2024 예측'])
        
        row = 0
        for hour in range(start_hour, 24):
            self.result_table.setItem(row, 0, QTableWidgetItem(f'{hour}시-{hour+1}시 사고 수'))
            for col, year in enumerate(['2019', '2020', '2021', '2022', '2023'], start=1):
                self.result_table.setItem(row, col, QTableWidgetItem(str(counts[year][hour])))
            self.result_table.setItem(row, 6, QTableWidgetItem(str(predictions_2024[hour])))
            row += 1

            self.result_table.setItem(row, 0, QTableWidgetItem(f'{hour}시-{hour+1}시 장소별 비율 (%)'))
            row += 1
            for place in ['교실', '교외', '부속시설', '운동장', '통로']:
                self.result_table.setItem(row, 0, QTableWidgetItem(place))
                for col, year in enumerate(['2019', '2020', '2021', '2022', '2023'], start=1):
                    self.result_table.setItem(row, col, QTableWidgetItem(f"{place_distribution[year][hour].get(place, 0):.2f}"))
                row += 1
        
        self.result_table.resizeColumnsToContents()
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

if __name__ == '__main__':
    file_path = 'c:\\Users\\kimdh\\Desktop\\공모전\\schooldata\\schoolData.xlsx'
    data = load_and_preprocess_data(file_path)
    
    app = QApplication(sys.argv)
    window = MainWindow(data)
    window.show()
    sys.exit(app.exec_())
