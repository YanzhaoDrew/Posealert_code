# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'childWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from posealert import posecompare, videocompare
from PyQt5 import QtCore, QtGui, QtWidgets


# 存储12个视频路径列表
inputpath_list = ["pic/test.jpg", "pic/zumba.mp4", "pic/walking.mp4"]
choice = 0


class Ui_DialogB1(object):
    def setupUi(self, Dialog, choice_End):
        global choice
        choice = choice_End
        Dialog.setObjectName("Dialog")
        Dialog.resize(1920, 1080)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget, 0, 0, 8, 2)
        self.tableWidget_2 = QtWidgets.QTableWidget(Dialog)
        self.tableWidget_2.setEnabled(True)
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(0)
        self.tableWidget_2.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget_2, 0, 2, 8, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_7 = QtWidgets.QPushButton(Dialog)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_3.addWidget(self.pushButton_7)
        self.pushButton_8 = QtWidgets.QPushButton(Dialog)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_3.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(Dialog)
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout_3.addWidget(self.pushButton_9)
        self.gridLayout.addLayout(self.verticalLayout_3, 1, 0, 1, 1)
        self.tableWidget_3 = QtWidgets.QTableWidget(Dialog)
        self.tableWidget_3.setObjectName("tableWidget_3")
        self.tableWidget_3.setColumnCount(0)
        self.tableWidget_3.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget_3, 0, 3, 8, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "ChildWindow"))
        self.pushButton_7.setText(_translate("Dialog", "Start"))
        self.pushButton_8.setText(_translate("Dialog", "nothing"))
        self.pushButton_9.setText(_translate("Dialog", "nothing"))
        self.pushButton_7.clicked.connect(self.start_train)

    def start_train(self):
        # 进行后缀判断,判断是静态图跟练，还是视频跟练
        bool_video = inputpath_list[choice].endswith(".mp4")
        if bool_video:
            videocompare(inputpath_list[choice])
        else:
            posecompare(inputpath_list[choice])
        self.close()
