# -*- coding: utf-8 -*-
from PyQt5.QtCore import Qt, QRect, QMetaObject, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QFont
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QPushButton, QRubberBand, QComboBox, QHBoxLayout, QVBoxLayout, QSpacerItem, QWidget, QMenuBar, \
    QStatusBar, QApplication, QLayout
import os

targetFolder = "./"
class Ui_MainWindow(object):
    fileName = ""
    fromLang = "en"
    toLang = "en"
    original = "./output.png"
    def setupUi(self, MainWindow):
        font = QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)

        #MainWindow.setStyleSheet("background-color: rgb(245, 245, 230);")

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(("centralwidget"))
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setFont(font)
        #self.verticalLayoutWidget.fitToWindowAct.setEnabled(True)
        self.verticalLayoutWidget.setGeometry(QRect(20, 20, 641, 571))
        self.verticalLayoutWidget.setObjectName(("verticalLayoutWidget"))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(("verticalLayout"))
        self.verticalLayout.setSizeConstraint(QLayout.SetMaximumSize)
        #self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(8)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.label_7 = QLabel(self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self.label_7)


        self.printer = QPrinter()
        self.scaleFactor = 0.0

        subclasses = self._subclass_container(self.verticalLayoutWidget)
        self.imageLabel = subclasses["rubberband"]
        #self.imageLabel = subclasses.RubberbandEnhancedLabel(self.verticalLayoutWidget)
        #self.imageLabel = QLabel(self.verticalLayoutWidget)
        #self.imageLabel.setGeometry(QRect(20, 20, 620, 300))
        self.imageLabel.setScaledContents(True)
        #self.imageLabel.setBackgroundRole(QPalette.Base)
        #self.imageLabel.setScaledContents(True)
        #
        # self.scrollArea = QScrollArea(self.verticalLayoutWidget)
        # self.scrollArea.setBackgroundRole(QPalette.Dark)
        # self.scrollArea.setWidget(self.imageLabel)
        # #self.scrollArea.setVisible(False)
        # #self.scrollArea.setMinimumSize(QSize(0, 350))
        # self.verticalLayout.addWidget(self.scrollArea)


        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.label.setText("From:")
        self.label.setFont(font)
        self.label.setMaximumSize(QSize(16777215, 30))
        self.verticalLayout.addWidget(self.label)

        self.comboBox = QComboBox(self.verticalLayoutWidget)
        self.comboBox.setObjectName(("comboBox"))
        self.comboBox.addItem("English")
        self.comboBox.addItem("Korean")
        self.comboBox.setFont(font)
        self.comboBox.currentIndexChanged.connect(self.selectionchange)
        self.verticalLayout.addWidget(self.comboBox)

        self.label_2 = QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(("label_2"))
        self.label_2.setText("To:")
        self.label_2.setFont(font)
        self.label_2.setMaximumSize(QSize(16777215, 30))
        self.verticalLayout.addWidget(self.label_2)

        self.comboBox_2 = QComboBox(self.verticalLayoutWidget)
        self.comboBox_2.setObjectName(("comboBox_2"))
        self.comboBox_2.addItem("English")
        self.comboBox_2.addItem("Korean")
        self.comboBox_2.setFont(font)
        self.comboBox_2.currentIndexChanged.connect(self.selectionchange2)
        self.verticalLayout.addWidget(self.comboBox_2)

        self.pushButton = QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName(("pushButton"))
        self.pushButton.setText("Translate")
        self.pushButton.setFont(font)
        self.pushButton.clicked.connect(self.changeImage)
        self.verticalLayout.addWidget(self.pushButton)

        spacerItem1 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.verticalLayoutWidget_2 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QRect(690, 20, 331, 571))

        MainWindow.setCentralWidget(self.centralwidget)



        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self.imageLabel)
        self.imageLabel.setMouseTracking(True)
        self.origin = QPoint()

        self.verticalLayoutWidget_2 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QRect(690, 20, 331, 571))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")

        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")


        self.label_3 = QLabel(self.verticalLayoutWidget_2)
        self.label_3.setMaximumSize(QSize(16777215, 30))
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_3.setText("Original:")
        self.verticalLayout_2.addWidget(self.label_3)

        self.label_5 = QLabel(self.verticalLayoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)

        self.label_4 = QLabel(self.verticalLayoutWidget_2)
        self.label_4.setMaximumSize(QSize(16777215, 30))
        self.label_4.setObjectName("label_4")
        self.label_4.setText("Translated:")
        self.label_4.setFont(font)
        self.verticalLayout_2.addWidget(self.label_4)

        self.label_6 = QLabel(self.verticalLayoutWidget_2)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)

        self.createActions()
        self.createMenus()

        #self.translate()
        # self.selectLang()


        #MainWindow.setObjectName(("Cartoon Translator"))
        MainWindow.resize(1000, 650)

        MainWindow.setWindowTitle("Cartoon Translator")
        # self.resize(800, 600)

        QMetaObject.connectSlotsByName(MainWindow)


    def open(self):
        #print("open!")
        options = QFileDialog.Options()
        #print("open!2")
        self.fileName, _ = QFileDialog.getOpenFileName(MainWindow, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)

        #print("open!3")
        if self.fileName:
            image = QImage(self.fileName)
            if image.isNull():
                QMessageBox.information(self.centralwidget, "Image Viewer", "Cannot load %s." % self.fileName)
                return

            #pm = QPixmap.fromImage(image.scaled(1000, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            pm = QPixmap.fromImage(image)
            #.scaledToHeight(200, Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(pm)
            #.scaledToWidth(500, Qt.KeepAspectRatio)

            self.scaleFactor = 1.0
            #self.scaleImage(1.25)

            #self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.imageLabel.setAlignment(Qt.AlignCenter)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def translate(self):
        btn = QPushButton("Translate", self)
        btn.setFont(QFont('Helvetica', 10))
        btn.resize(100, 50)
        btn.move(350, 400)
        btn.setContentsMargins(0, 0, 0, 0)
        btn.clicked.connect(self.changeImage)
        #self.verticalLayout.addWidget(btn, alignment=Qt.AlignCenter)
        #btn.setTextAlignment(Qt.AlignCenter)
        #self.show()

    def selectionchange(self):
        if (self.comboBox.currentText() == "Korean"):
            self.fromLang = "ko"

    def selectionchange2(self):
        if (self.comboBox_2.currentText() == "Korean"):
            self.toLang = "ko"

    def changeImage(self):
        print(self.fileName)
        #print(type(self.fileName))
        os.system(f'python main.py -i output.png --lang {self.fromLang}')
        #os.system(f'python read_png_refactored.py -i {self.fileName} -m mask.png -s {self.fromLang} -t {self.toLang}')

        image = QImage("./translated.jpg")
        pm = QPixmap.fromImage(image.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label_6.setPixmap(pm)
        self.scaleFactor = 1.0

        self.label_6.setAlignment(Qt.AlignCenter)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.label_6.adjustSize()

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        #self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self.centralwidget, "About Cartoon Translator","       Developers:\n       Woo Jae Kim\n      Wonho Zhung\n      Chae Won Kim")

    def createActions(self):
        self.openAct = QAction("&Open...", self.centralwidget, shortcut="Ctrl+O", triggered=self.open)
        self.printAct = QAction("&Print...", self.centralwidget, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QAction("E&xit", self.centralwidget, shortcut="Ctrl+Q", triggered=self.centralwidget.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self.centralwidget, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self.centralwidget, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self.centralwidget, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self.centralwidget, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self.centralwidget, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self.centralwidget, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self.centralwidget)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self.centralwidget)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self.centralwidget)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        MainWindow.menuBar().addMenu(self.fileMenu)
        MainWindow.menuBar().addMenu(self.viewMenu)
        MainWindow.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        # self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        # self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)
    #
    # def adjustScrollBar(self, scrollBar, factor):
    #     scrollBar.setValue(int(factor * scrollBar.value()
    #                            + ((factor - 1) * scrollBar.pageStep() / 2)))

    def _subclass_container(self, QWidget):
        _parent_class = self
        class RubberbandEnhancedLabel(QLabel):

            def __init__(self, parent=None):
                self._parent_class = _parent_class
                QLabel.__init__(self, parent)
                self.selection = QRubberBand(QRubberBand.Rectangle, self)

            def mousePressEvent(self, event):
                '''
                    Mouse is pressed. If selection is visible either set dragging mode (if close to border) or hide selection.
                    If selection is not visible make it visible and start at this point.
                '''
                #
                if event.button() == Qt.LeftButton:

                    position = QPoint(event.pos())
                    if self.selection.isVisible():
                        # visible selection
                        if (self.upper_left - position).manhattanLength() < 20:
                            # close to upper left corner, drag it
                            self.mode = "drag_upper_left"
                        elif (self.lower_right - position).manhattanLength() < 20:
                            # close to lower right corner, drag it
                            self.mode = "drag_lower_right"
                        else:
                            # clicked somewhere else, hide selection
                            self.selection.hide()
                    else:
                        # no visible selection, start new selection
                        self.upper_left = position
                        self.lower_right = position
                        self.mode = "drag_lower_right"
                        self.selection.show()

                    # self.originQPoint = eventQMouseEvent.pos()
                    # self.currentQRubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)
                    # self.selection.setGeometry(QRect(self.originQPoint, QSize()))
                    # self.selection.show()

            def mouseMoveEvent(self, event):
                '''
                    Mouse moved. If selection is visible, drag it according to drag mode.
                '''
                if self.selection.isVisible():
                    # visible selection
                    if self.mode == "drag_lower_right":
                        self.lower_right = QPoint(event.pos())
                    elif self.mode == "drag_upper_left":
                        self.upper_left = QPoint(event.pos())
                    # update geometry
                    self.selection.setGeometry(QRect(self.upper_left, self.lower_right).normalized())

            def mouseReleaseEvent(self, event):
                # if event.button() == Qt.LeftButton:
                #     self.rubberBand.hide()
                if self.selection.isVisible():
                    currentQRect = self.selection.geometry()
                    #self.selection.deleteLater()
                    cropQPixmap = self.pixmap().copy(currentQRect)
                    cropQPixmap.save('output.png')

                    image = QImage("output.png")
                    #print("crop2")
                    if image.isNull():
                        QMessageBox.information(self.centralwidget, "Image Viewer", "Cannot load %s." % self.fileName)
                        return

                    #print("crop3")
                    pm = QPixmap.fromImage(image)
                    # pm = QPixmap.fromImage(image)

                    self._parent_class.label_5.setPixmap(pm)
                    #self.label_5.setPixmap(pm)
                    #self.label_5.setAlignment(Qt.AlignCenter)

                    self.selection.hide()

            # def resizeEvent(self, event):
            #     size = QSize(1, 1)
            #     size.scale(self.size(), Qt.KeepAspectRatio)
            #     self.resize(size)
            #     self.selection.resize(self.size())

        return {"rubberband": RubberbandEnhancedLabel(self.verticalLayoutWidget)}



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
