import sys
from posealert import posecompare
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox

class AI_Fitness_UI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Fitness")

        button1 = QPushButton("Yoga")
        button2 = QPushButton("Hit")

        layout = QVBoxLayout()
        layout.addWidget(button1)
        layout.addWidget(button2)
        self.setLayout(layout)

        button1.clicked.connect(self.posewin)
        button2.clicked.connect(self.fitwin)
        self.mainwin()

    def mainwin(self):
        self.show()

    def posewin(self):
        inputpath = "pic/test.jpg"
        posecompare(inputpath)
        self.hide()

    def fitwin(self):
        inputpath = "pic/video.mp4"
        posecompare(inputpath)
        self.hide()


class PoseAlert(QWidget):
    def __init__(self, inputpath):
       super().__init__()
       self.setWindowTitle("Result")
       label = QLabel("targte file is {}".format(inputpath))
       button = QPushButton("OK")
       layout = QVBoxLayout()
       layout.addWidget(label)
       layout.addWidget(button)
       self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = AI_Fitness_UI()
    sys.exit(app.exec_())