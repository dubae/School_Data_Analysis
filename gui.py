import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout
from pages.AccidentLocationPage import AccidentLocationPage, load_and_preprocess_data  # AccidentLocationPage 클래스와 데이터 로드 함수 임포트
from pages.AccidentPartPage import AccidentPartPage
from pages.AccidentTypePage import AccidentTypePage
from pages.AccidentActivityPage import AccidentActivityPage
from pages.AccidentObjectPage import AccidentObjectPage
from pages.AccidentGradePage import AccidentGradePage
# 메인 윈도우
class MainWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()

        self.data = data

        self.setWindowTitle("학교 안전사고 예측 프로그램")
        self.setGeometry(100, 100, 1200, 800)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.main_layout = QHBoxLayout(self.main_widget)

        # 버튼 레이아웃 설정
        self.button_layout = QVBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # 가장자리 여백 제거
        self.button_layout.setSpacing(0)  # 버튼 간격 제거
        self.main_layout.addLayout(self.button_layout)

        # 우측 페이지 스택 위젯 설정
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # 페이지 설정
        self.pages = {
            "사고장소": AccidentLocationPage(self.data),
            "사고부위": AccidentPartPage(self.data),
            "사고형태": AccidentTypePage(self.data),
            "사고당시활동": AccidentActivityPage(self.data),
            "사고매개물": AccidentObjectPage(self.data),
            "사고학년": AccidentGradePage(self.data)
        }

        # 버튼 및 페이지 추가
        for label, page in self.pages.items():
            button = QPushButton(label)
            button.clicked.connect(lambda checked, lbl=label: self.display_page(lbl))
            self.button_layout.addWidget(button)

            self.stacked_widget.addWidget(page)

        # 기본 페이지 설정
        self.display_page("사고장소")

    def display_page(self, label):
        page = self.pages.get(label)
        if page:
            self.stacked_widget.setCurrentWidget(page)

if __name__ == '__main__':
    file_path = 'schoolData.xlsx'
    data = load_and_preprocess_data(file_path)
    
    app = QApplication(sys.argv)
    window = MainWindow(data)
    window.show()
    sys.exit(app.exec_())