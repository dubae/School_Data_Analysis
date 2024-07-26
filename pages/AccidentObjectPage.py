from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class AccidentObjectPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 페이지별 정보 추가
        label = QLabel("이 페이지는 사고매개물에 대한 정보를 표시합니다.")
        layout.addWidget(label)
