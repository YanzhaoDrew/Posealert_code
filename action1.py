import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
# from Mainwindow8_former import Ui_MainWindow
from Mainwindow8 import Ui_MainWindow
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # create the main window UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
