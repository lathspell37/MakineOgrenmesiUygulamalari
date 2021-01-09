from PyQt5 import QtWidgets
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

#Screens
import RandomGridSearch



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = RandomGridSearch.Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
