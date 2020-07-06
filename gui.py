# -*- coding: utf-8 -*-
import os

from PyQt5.QtCore import Qt, QRect, QMetaObject, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QPushButton, QRubberBand, QComboBox, QVBoxLayout, QSpacerItem, QWidget, QApplication, QLayout

targetFolder = "./"
class Ui_MainWindow(object):
    fileName = ""
    fromLang = "en"
    toLang = "en"
    original = "./output.png"
    max_h = 600
    max_w = 1100

    def setupUi(self, MainWindow):
        MainWindow.showMaximized()
        
        # Default font
        font = QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)

        self.centralwidget = QWidget(MainWindow)
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setFont(font)
        self.verticalLayoutWidget.setGeometry(QRect(20, 60, 1100, 891))
        
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QLayout.SetMaximumSize)
        self.verticalLayout.setSpacing(8)

        # for spacing UI
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.outputLabel = QLabel(self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self.outputLabel)

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        # To select area
        subclasses = self._subclass_container(self.verticalLayoutWidget)
        self.imageLabel = subclasses["rubberband"]

        # Select From Language
        self.fromLabel = QLabel(self.verticalLayoutWidget)
        self.fromLabel.setText("From:")
        self.fromLabel.setFont(font)
        self.fromLabel.setMaximumSize(QSize(16777215, 30))
        self.verticalLayout.addWidget(self.fromLabel)

        self.comboBox = QComboBox(self.verticalLayoutWidget)
        self.comboBox.addItem("English")
        self.comboBox.addItem("Korean")
        self.comboBox.setFont(font)
        self.comboBox.currentIndexChanged.connect(self.selectionchange)
        self.verticalLayout.addWidget(self.comboBox)

        # Select To Language
        self.toLabel = QLabel(self.verticalLayoutWidget)
        self.toLabel.setText("To:")
        self.toLabel.setFont(font)
        self.toLabel.setMaximumSize(QSize(16777215, 30))
        self.verticalLayout.addWidget(self.toLabel)

        self.comboBox_2 = QComboBox(self.verticalLayoutWidget)
        self.comboBox_2.addItem("English")
        self.comboBox_2.addItem("Korean")
        self.comboBox_2.setFont(font)
        self.comboBox_2.currentIndexChanged.connect(self.selectionchange2)
        self.verticalLayout.addWidget(self.comboBox_2)

        # Translate Button
        self.translateButton = QPushButton(self.verticalLayoutWidget)
        self.translateButton.setText("Translate")
        self.translateButton.setFont(font)
        self.translateButton.clicked.connect(self.changeImage)
        self.verticalLayout.addWidget(self.translateButton)

        spacerItem1 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)

        # Instruction text label
        self.instructutionLabel = QLabel(self.centralwidget)
        self.instructutionLabel.setGeometry(QRect(60, 20, 651, 16))
        self.instructutionLabel.setFont(font)
        self.instructutionLabel.setText("*Warning: Please select individual scene only, not the speech bubble!*")


        MainWindow.setCentralWidget(self.centralwidget)

        
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self.imageLabel)
        self.imageLabel.setMouseTracking(True)
        self.origin = QPoint()

        self.verticalLayoutWidget_2 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QRect(1200, 60, 591, 900))

        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)

        # Original image text label
        self.originalLabel = QLabel(self.verticalLayoutWidget_2)
        self.originalLabel.setMaximumSize(QSize(16777215, 30))
        self.originalLabel.setFont(font)
        self.originalLabel.setText("Original:")
        self.verticalLayout_2.addWidget(self.originalLabel)

        # Original image label
        self.originalImageLabel = QLabel(self.verticalLayoutWidget_2)
        self.originalImageLabel.setObjectName("originalImageLabel")
        self.originalImageLabel.setMinimumSize(QSize(0, 300))
        self.verticalLayout_2.addWidget(self.originalImageLabel)

        # Translated image text label
        self.translatedLabel = QLabel(self.verticalLayoutWidget_2)
        self.translatedLabel.setMaximumSize(QSize(16777215, 30))
        self.translatedLabel.setText("Translated:")
        self.translatedLabel.setFont(font)
        self.verticalLayout_2.addWidget(self.translatedLabel)

        # Translated image label
        self.translatedImageLabel = QLabel(self.verticalLayoutWidget_2)
        self.translatedImageLabel.setMinimumSize(QSize(0, 300))
        self.verticalLayout_2.addWidget(self.translatedImageLabel)

        #Menu
        self.createActions()
        self.createMenus()

        MainWindow.setWindowTitle("Cartoon Translator")
        QMetaObject.connectSlotsByName(MainWindow)


    def open(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(MainWindow, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)

        if self.fileName:
            image = QImage(self.fileName)
            if image.isNull():
                QMessageBox.information(self.centralwidget, "Image Viewer", "Cannot load %s." % self.fileName)
                return

            pm = QPixmap.fromImage(image)
            w = pm.width()
            h = pm.height()

            if (w <= self.max_w and h <= self.max_h):
                self.imageLabel.setPixmap(pm)
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Image Size Error")
                msg.setText("Image size is too big!")
                x = msg.exec_()  # this will show our messagebox


            self.scaleFactor = 1.0

            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.imageLabel.setAlignment(Qt.AlignCenter)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()


    def selectionchange(self):
        if (self.comboBox.currentText() == "Korean"):
            self.fromLang = "ko"

    def selectionchange2(self):
        if (self.comboBox_2.currentText() == "Korean"):
            self.toLang = "ko"

    def changeImage(self):
        os.system(f'python main.py -i output.png --lang {self.fromLang}')

        image = QImage("./translated.jpg")
        pm = QPixmap.fromImage(image)

        w = pm.width()
        h = pm.height()

        if (h > w):
            self.translatedImageLabel.setPixmap(pm.scaledToHeight(400))

        else:
            self.translatedImageLabel.setPixmap(pm.scaledToWidth(400))

        self.scaleFactor = 1.0

        self.updateActions()

        os.remove('./translated.jpg')
        os.remove('./output.png')

        if not self.fitToWindowAct.isChecked():
            self.translatedImageLabel.adjustSize()

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
        # MainWindow.menuBar().addMenu(self.viewMenu)
        MainWindow.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())


        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)


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

                if self.selection.isVisible():
                    currentQRect = self.selection.geometry()
                    cropQPixmap = self.pixmap().copy(currentQRect)
                    cropQPixmap.save('output.png')

                    image = QImage("output.png")
                    if image.isNull():
                        QMessageBox.information(self.centralwidget, "Image Viewer", "Cannot load %s." % self.fileName)
                        return

                    pm = QPixmap.fromImage(image)
                    h = pm.height()
                    w = pm.width()

                    if (h > w):
                        self._parent_class.originalImageLabel.setPixmap(pm.scaledToHeight(400))

                    else:
                        self._parent_class.originalImageLabel.setPixmap(pm.scaledToWidth(400))



                    self.selection.hide()


        return {"rubberband": RubberbandEnhancedLabel(self.verticalLayoutWidget)}



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
