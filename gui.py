from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from main import main


class ProgramInfo():
    def __init__(self):
        self.close = False
        self.preferRightHand = True

    def setIsRightHandPreferred(self, val):
        self.preferRightHand = val

    def getIsRightHandPreferred(self):
        return self.preferRightHand

    def setClose(self, v):
        self.close = v

    def closeDetecting(self):
        return self.close


info = ProgramInfo()


class SecondWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(SecondWindow, self).__init__()
        self.main_widget = QtWidgets.QWidget(self)
        self.setWindowTitle("AI VM")
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.setStyleSheet("background-color:#2155CD; padding:5px;")

        workModeButton = QtWidgets.QPushButton("Stop", self)
        workModeButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        workModeButton.setObjectName("stop")
        workModeButton.resize(120, 50)
        workModeButton.setStyleSheet(
            "background-color:#fff; font-size:25px; border-radius:3px; font-weight:500")
        workModeButton.move(40, 25)

        workModeButton.clicked.connect(self.clicked)

        QtWidgets.QVBoxLayout(self.main_widget)

    def clicked(self):
        MainWindow.show()
        info.setClose(True)
        self.close()


class Ui_MainWindow(object):

    def location_on_the_screen(self):
        ag = QtWidgets.QDesktopWidget().availableGeometry()

        widget = self.SW.geometry()
        x = ag.width() - widget.width() - 20
        y = widget.height() // 2 + 2
        self.SW.move(x, y)

    def clicked(self):
        MainWindow.hide()
        info.setClose(False)
        self.SW = SecondWindow()
        self.SW.resize(200, 100)
        self.location_on_the_screen()
        self.SW.show()
        info.setIsRightHandPreferred(self.rightHand.checkState())
        main(info)

    def uncheck(self, state):
        if state == True:
            if self.centralwidget.sender() == self.rightHand:
                self.leftHand.setChecked(False)
            elif self.centralwidget.sender() == self.leftHand:
                self.rightHand.setChecked(False)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)
        MainWindow.setStyleSheet("#MainWindow{\n"
                                 "    background-color:#2155CD;\n"
                                 "}\n"
                                 "#mainFrame{\n"
                                 "    width:100%;\n"
                                 "    padding:15px\n"
                                 "}\n"
                                 "#MMvalues\n"
                                 "{\n"
                                 "    padding:0\n"
                                 "}\n"
                                 "#program_title\n"
                                 "{\n"
                                 "    color: #fff;\n"
                                 "    font-size:50px;\n"
                                 "    font-weight:600;\n"
                                 "}\n"
                                 "QLabel{\n"
                                 "    color:#fff;\n"
                                 "    font-size:20px;\n"
                                 "    margin:20px 0px 8px\n"
                                 "}\n"
                                 "#plotButton\n"
                                 "{\n"
                                 "    background-color:#fff;\n"
                                 "    font-size: 25px;\n"
                                 "    border-radius:3px;\n"
                                 "    padding: 13px 0;\n"
                                 "}\n"
                                 "QLineEdit\n"
                                 "{\n"
                                 "    padding: 14px 16px;\n"
                                 "    font-size: 15px;\n"
                                 "    border-radius:3px;\n"
                                 "}\n"
                                 "#leftHand{\n"
                                 "    margin-bottom:35px\n"
                                 "}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.mainFrame = QtWidgets.QFrame(self.centralwidget)
        self.mainFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainFrame.setObjectName("mainFrame")
        self.formLayout = QtWidgets.QFormLayout(self.mainFrame)
        self.formLayout.setObjectName("formLayout")

        self.label_2 = QtWidgets.QLabel(self.mainFrame)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(
            5, QtWidgets.QFormLayout.LabelRole, self.label_2)

        self.rightHand = QtWidgets.QCheckBox(self.mainFrame)
        self.rightHand.setObjectName("rightHand")
        self.formLayout.setWidget(
            6, QtWidgets.QFormLayout.SpanningRole, self.rightHand)

        self.leftHand = QtWidgets.QCheckBox(self.mainFrame)
        self.leftHand.setObjectName("leftHand")
        self.formLayout.setWidget(
            8, QtWidgets.QFormLayout.SpanningRole, self.leftHand)

        self.plotButton = QtWidgets.QPushButton(self.mainFrame)
        self.plotButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.plotButton.setObjectName("plotButton")
        self.formLayout.setWidget(
            9, QtWidgets.QFormLayout.SpanningRole, self.plotButton)
        self.program_title = QtWidgets.QLabel(self.mainFrame)
        self.program_title.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.program_title.setAlignment(QtCore.Qt.AlignCenter)
        self.program_title.setObjectName("program_title")
        self.formLayout.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.program_title)
        self.gridLayout.addWidget(self.mainFrame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.plotButton.clicked.connect(self.clicked)

        self.rightHand.toggled.connect(self.uncheck)
        self.leftHand.toggled.connect(self.uncheck)
        # self.leftHand.stateChanged.connect(self.uncheck(self.leftHand))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowIcon(QtGui.QIcon('icon.png'))
        MainWindow.setWindowTitle(_translate("MainWindow", "AI Virtual Mouse"))
        checkboxStyle = "color:#fff;font-weight:500;margin-left:25px;"
        self.label_2.setText(_translate(
            "MainWindow", "Select The Desired Hand"))
        self.rightHand.setText("Right Hand")
        self.rightHand.setStyleSheet(
            checkboxStyle+"margin-bottom:10px;margin-top:2px")
        self.rightHand.setChecked(True)

        self.leftHand.setText("Left Hand")
        self.leftHand.setStyleSheet(checkboxStyle)
        self.plotButton.setText(_translate("MainWindow", "Start"))
        self.plotButton.setStyleSheet("font-weight:600;color:#111")
        self.program_title.setText(_translate("MainWindow", "Virtual Mouse"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
