import os
from threading import *

import cv2
import tensorflow as tf
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from deepface import DeepFace

from ui_mainWindow import Ui_MainWindow

os.environ["CUDA_DEVICE_ORDER"] = "0000:01:00.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def highlightFace(net, frame, conf_threshold=0.6):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


class Analyze(Thread):
    def __init__(self, ui):
        Thread.__init__(self)
        self.image = None
        self.ui = ui
        self.fileName = None
        self.response = None

    def run(self):
        print(active_count())
        self.fileName = f'image/0.png'
        cv2.imwrite(self.fileName, self.image)
        self.response = DeepFace.analyze(self.image)
        self.UI_Set()

    def UI_Set(self):
        self.ui.lbl_cinsiyet.clear()
        self.ui.lbl_duygu.clear()
        self.ui.lbl_yas.clear()
        self.ui.lbl_irk.clear()
        if self.ui.cins_check.isChecked():
            self.ui.lbl_cinsiyet.setText(str(self.response["gender"]))
        if self.ui.duygu_check.isChecked():
            self.ui.lbl_duygu.setText(str(self.response["dominant_emotion"]))
        if self.ui.yas_check.isChecked():
            self.ui.lbl_yas.setText(str(round(self.response["age"])))
        if self.ui.irk_check.isChecked():
            self.ui.lbl_irk.setText(str(self.response["dominant_race"]))

        self.ui.lbl_foto.setPixmap(QPixmap(self.fileName))
        self.ui.lbl_foto.setScaledContents(True)
        tf.keras.backend.clear_session()

    def getImage(self, image):
        self.image = image


class YapayZekaProjeWidget(QMainWindow):

    def __init__(self, parent=None):
        super(YapayZekaProjeWidget, self).__init__(parent=parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(931, 675)
        self.initSlots()
        self.functionsUnit()
        self.analiz = Analyze(self.ui)
        self.ui.btn_startAnalysis.setEnabled(False)
        self.ui.actionStop_Cam.setEnabled(False)
        self.ui.btn_ayar.setEnabled(False)

    def analyze(self):
        QMessageBox.information(self, "Bilgi", "Analiz Başlıyor...")
        self.analiz.getImage(self.copyImage)
        self.analiz.run()

    def functionsUnit(self):
        faceProto = "models/opencv_face_detector.pbtxt"
        faceModel = "models/opencv_face_detector_uint8.pb"
        ageProto = "models/age_deploy.prototxt"
        ageModel = "models/age_net.caffemodel"
        genderProto = "models/gender_deploy.prototxt"
        genderModel = "models/gender_net.caffemodel"

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Erkek', 'Kadın']
        self.padding = 20
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
        self.copyImage = None
        self.input_shape = (224, 224)

    def initSlots(self):
        self.kontrol = False
        self.ui.actionResim_Ekle.triggered.connect(self.getImageFromFile)
        self.ui.btn_ayar.clicked.connect(self.loadModels)
        self.ui.actionStart_Cam.triggered.connect(self.startCam)
        self.ui.actionStop_Cam.triggered.connect(self.stopWebCam)
        self.ui.btn_startAnalysis.clicked.connect(self.analyze)
        self.ui.actionExit.triggered.connect(lambda: QtCore.QCoreApplication.instance().quit())

    def startCam(self):
        self.ui.btn_startAnalysis.setEnabled(True)
        self.ui.actionStop_Cam.setEnabled(True)
        self.capture = cv2.VideoCapture(0)
        self.Kontrol = True
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 607)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 576)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000. / 24)

    def update_frame(self):
        ret, self.frame = self.capture.read()
        self.copyImage = self.frame.copy()
        resultImg, faceBoxes = highlightFace(self.faceNet, self.frame)

        for faceBox in faceBoxes:
            face = self.frame[max(0, faceBox[1] - self.padding):
                              min(faceBox[3] + self.padding,
                                  self.frame.shape[0] - 1),
                   max(0, faceBox[0] - self.padding)
                   :min(faceBox[2] + self.padding, self.frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = self.genderList[genderPreds[0].argmax()]
            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[agePreds[0].argmax()]
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        self.displayImage(resultImg)

    def displayImage(self, img):
        qFormat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qFormat = QImage.Format_RGBA8888
            else:
                qFormat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qFormat)
        # BGR to RGB
        self.outImage = outImage.rgbSwapped()
        self.ui.label.setPixmap(QPixmap.fromImage(self.outImage))
        self.ui.label.setScaledContents(True)

    def stopWebCam(self):
        self.capture.release()
        self.timer.stop()

    def getImageFromFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "PNG Files (*.png);;JPG Files (*.jpg) ",
                                                  options=options)
        if fileName:
            print(fileName)
            self.resim = str(fileName)
            self.loadImage(fileName)
            self.ui.btn_ayar.setEnabled(True)
        else:
            self.resimCheck = False

    def loadImage(self, fileName):
        self.ImageStudentPath = fileName
        self.ui.lbl_foto.setPixmap(QPixmap(self.ImageStudentPath))
        self.ui.lbl_foto.setScaledContents(True)

    def loadModels(self):
        self.ui.lbl_cinsiyet.clear()
        self.ui.lbl_duygu.clear()
        self.ui.lbl_yas.clear()
        self.ui.lbl_irk.clear()
        QMessageBox.information(self, "Bilgi", "Analiz Başlıyor...")
        path = self.resim
        img = cv2.imread(path)
        resp = DeepFace.analyze(img)
        if self.ui.cins_check.isChecked():
            self.ui.lbl_cinsiyet.setText(str(resp["gender"]))
        if self.ui.duygu_check.isChecked():
            self.ui.lbl_duygu.setText(str(resp["dominant_emotion"]))
        if self.ui.yas_check.isChecked():
            self.ui.lbl_yas.setText(str(round(resp["age"])))
        if self.ui.irk_check.isChecked():
            self.ui.lbl_irk.setText(str(resp["dominant_race"]))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    t = Thread(target=YapayZekaProjeWidget)
    t.start()
    mainWindow = YapayZekaProjeWidget()
    t.join()
    mainWindow.show()
    sys.exit(app.exec_())
