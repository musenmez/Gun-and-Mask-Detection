# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:33:31 2020

@author: Monster
"""

# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QThread
from pynput.keyboard import Key, Controller

# import Opencv module
import cv2


from ui_template import *
from main import *

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()        
        self.ui.setupUi(self)
        self.model = 1;
        
        self.keyboard = Controller()
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.show_frame)
        # set control_bt callback clicked  function
        self.ui.streamControl.clicked.connect(self.controlTimer)        
        #self.show_popup()
        self.totalCount = 0
        self.ppimage = "gui_images/pp.jpeg"       
        
    
    def get_value(self):
        return self.ui
        
    def show_frame(self):     
        self.error = get_error()
        if self.error == 404:            
            self.controlTimer()
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Invalid camera number")        
            x = msg.exec_()  
            return -1       
        
        if self.error == 101:            
            self.controlTimer()
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Invalid token")        
            x = msg.exec_()  
            return -1
            
            
        image, self.cap, faceList, mask_status = get_all_values()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label        
        self.ui.image_label.setPixmap(QtGui.QPixmap(qImg))
        
        if faceList != None:
            count = 0
            totalHolder = 0
            for i in range(len(faceList)): #range(len(faceList))
                if faceList[i] is None:  
                    continue
                else:                    
                    cf = cv2.cvtColor(faceList[i], cv2.COLOR_BGR2RGB)
                    # get image infos
                    height, width, channel = cf.shape
                    step = channel * width
                    # create QImage from image
                    qImg = QImage(cf.data, width, height, step, QImage.Format_RGB888)
                    # show image in img_label        
                    self.control_faces(i,count,qImg,mask_status[i])
                    count = count + 1
                    totalHolder= i
                    
            for i in range(count, 6):
                if i == 0:                    
                    self.ui.pp0.setPixmap(QtGui.QPixmap(self.ppimage)) 
                    self.ui.id0.setText("")
                elif i == 1:
                    self.ui.pp1.setPixmap(QtGui.QPixmap(self.ppimage)) 
                    self.ui.id1.setText("")
                elif i == 2:
                    self.ui.pp2.setPixmap(QtGui.QPixmap(self.ppimage)) 
                    self.ui.id2.setText("")
                elif i == 3:
                    self.ui.pp3.setPixmap(QtGui.QPixmap(self.ppimage)) 
                    self.ui.id3.setText("")
                elif i == 4:
                    self.ui.pp4.setPixmap(QtGui.QPixmap(self.ppimage)) 
                    self.ui.id4.setText("")
                elif i == 5:
                    self.ui.pp5.setPixmap(QtGui.QPixmap(self.ppimage)) 
                    self.ui.id5.setText("")
                else:
                    print("Out of limit")
                
            self.ui.currentCustomers.setText(str(count))
            
            if totalHolder >= self.totalCount:
                self.totalCount = totalHolder + 1
                self.ui.totalCustomers.setText(str(self.totalCount))
    
    
        
    def show_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("Invalid camera number")        
        x = msg.exec_()    

   
    # start/stop timer
    def controlTimer(self):            
        MainWindow.cameraID = int(self.ui.cameraID.text())
        MainWindow.chatID = int(self.ui.chatID.text())  
        MainWindow.token = self.ui.token.text()
        
        if self.ui.gunCheckBox.isChecked():
            MainWindow.model = 1
        elif self.ui.maskCheckBox.isChecked():
            MainWindow.model = 2
        elif self.ui.gunMaskCheckBox.isChecked():
            MainWindow.model = 3
        
        self.threadClass = ThreadClass()
        
        # if timer is stopped       
        
        if not self.timer.isActive():        
            print("start timer")
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.streamControl.setText("DURDUR")                 
            self.threadClass.start()    
            self.resetData()
        # if timer is started
        else:
            print("stop timer")
            # stop timer            
            self.timer.stop()  
            # update control_bt text
            self.ui.streamControl.setText("Start Stream")
            self.resetData()
            check_stop(1)

    def control_faces(self, IDnum, count, image, mask_status):
        if mask_status == 1:
            mask_text = "Var"
        else:
            mask_text = "Yok"
            
        if count == 0:
            self.ui.pp0.setPixmap(QtGui.QPixmap(image)) 
            self.ui.id0.setText("ID_" + str(IDnum))
            self.ui.mask0.setText(mask_text)
        elif count == 1:
            self.ui.pp1.setPixmap(QtGui.QPixmap(image)) 
            self.ui.id1.setText("ID_" + str(IDnum))
            self.ui.mask1.setText(mask_text)
        elif count == 2:
            self.ui.pp2.setPixmap(QtGui.QPixmap(image)) 
            self.ui.id2.setText("ID_" + str(IDnum))
            self.ui.mask2.setText(mask_text)
        elif count == 3:
            self.ui.pp3.setPixmap(QtGui.QPixmap(image)) 
            self.ui.id3.setText("ID_" + str(IDnum))
            self.ui.mask3.setText(mask_text)
        elif count == 4:
            self.ui.pp4.setPixmap(QtGui.QPixmap(image)) 
            self.ui.id4.setText("ID_" + str(IDnum))
            self.ui.mask4.setText(mask_text)
        elif count == 5:
            self.ui.pp5.setPixmap(QtGui.QPixmap(image)) 
            self.ui.id5.setText("ID_" + str(IDnum))
            self.ui.mask5.setText(mask_text)
        else:
            print("Out of limit")     
            
    def resetData(self):       
        
        
        #Reset Images
        self.ui.image_label.setPixmap(QtGui.QPixmap("gui_images/cctvholder.png"))		
        self.ui.pp0.setPixmap(QtGui.QPixmap("gui_images/pp.jpeg"))
        self.ui.pp1.setPixmap(QtGui.QPixmap("gui_images/pp.jpeg"))            
        self.ui.pp2.setPixmap(QtGui.QPixmap("gui_images/pp.jpeg"))
        self.ui.pp3.setPixmap(QtGui.QPixmap("gui_images/pp.jpeg"))
        self.ui.pp4.setPixmap(QtGui.QPixmap("gui_images/pp.jpeg"))
        self.ui.pp5.setPixmap(QtGui.QPixmap("gui_images/pp.jpeg"))
        
        #Reset ID
        self.ui.id0.setText("")
        self.ui.id1.setText("")
        self.ui.id2.setText("")
        self.ui.id3.setText("")
        self.ui.id4.setText("")
        self.ui.id5.setText("")
        
        #Reset Counters
        self.totalCount = 0
        self.ui.currentCustomers.setText("0")   
        self.ui.totalCustomers.setText("0")
        

        
class ThreadClass(QtCore.QThread):
    def __init__(self, parent = None):
        super(ThreadClass, self).__init__(parent)  
        self.MainW = MainWindow()   
    
    def run(self):      
        print("run")    
        dm = main(MainWindow.cameraID, MainWindow.chatID, MainWindow.token, MainWindow.model)  
        self.check_error(dm)  
        
    def check_error(self, error):
        if error == 404:
            print("hello") 
            
            
   
            
    

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())

