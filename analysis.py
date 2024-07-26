import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTextEdit
import datetime
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
def get_accident_counts_and_place_distribution(data, region, day, start_hour, end_hour):
    counts = {}
    place_distribution = {}
    place_counts_total = {}
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

        # 해당 지역, 요일 및 시간대 필터링
        filtered_df = df_copy[(df_copy['지역'] == region) & (df_copy['사고발생요일'] == day) & (df_copy['사고발생시각'] >= start_hour) & (df_copy['사고발생시각'] < end_hour)]
        
        # 사고 수 계산
        counts[year] = len(filtered_df)
        
        # 사고 장소별 비율 및 횟수 계산
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
        
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
    
    def show_accident_counts(self):
        region = self.region_input.text()
        day = self.day_input.text()
        start_hour = int(self.start_hour_input.text())
        end_hour = int(self.end_hour_input.text())
        
        counts, place_distribution, place_counts_total = get_accident_counts_and_place_distribution(self.data, region, day, start_hour, end_hour)
        predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_place(self.data, region, day, start_hour, end_hour)
        
        result_text = f"{region} 지역에서 {day}요일 {start_hour}시부터 {end_hour}시까지의 사고 수:\n"
        for year, count in counts.items():
            result_text += f"{year}: {count}건\n"
        
        result_text += "\n사고 장소별 비율 및 횟수:\n"
        for year, distribution in place_distribution.items():
            result_text += f"{year}년:\n"
            for place, percentage in distribution.items():
                count = place_counts_total[year].get(place, 0)
                result_text += f"  {place}: {count}건 ({percentage:.2f}%)\n"
        
        result_text += f"\n2024년 {day}요일에 예측된 사고 수:\n"
        for place, count in predicted_counts_2024.items():
            percentage = predicted_percentage_2024[place]
            result_text += f"  {place}: {count:.2f}건 ({percentage:.2f}%)\n"
        result_text += f"\n총 예측 사고 수: {total_predicted_count:.2f}건\n"
        
        # 여기서부터는 두회의 테스트
        result_text += f"\n--------------TEST--------------\n"
        for i in range(start_hour,24,1):
            predicted_counts_2024, total_predicted_count, predicted_percentage_2024 = predict_accidents_by_place(self.data, region, day, i, i+1)
            result_text += f"\n2024년 {day}요일 {i}~{i+1}에 예측된 사고 수:\n"
            for place, count in predicted_counts_2024.items():
                percentage = predicted_percentage_2024[place]
                result_text += f"  {place}: {count:.2f}건 ({percentage:.2f}%)\n"
            result_text += f"\n총 예측 사고 수: {total_predicted_count:.2f}건\n"
        
        


        
        self.result_label.setText(result_text)

if __name__ == '__main__':
    file_path = 'schoolData.xlsx'
    data = load_and_preprocess_data(file_path)
    
    app = QApplication(sys.argv)
    window = MainWindow(data)
    window.show()
    sys.exit(app.exec_())
