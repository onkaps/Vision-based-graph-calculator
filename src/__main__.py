import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from .snipping_tool import SnippingWidget
from .model.train import train_model
from .model.test import test_model
from .utils import BUILD_DIR
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Screen Capture Tool')
        self.setGeometry(100, 100, 300, 200)

        captureBtn = QPushButton('Capture Screen', self)
        captureBtn.move(100, 80)
        captureBtn.clicked.connect(self.start_snipping)

    def start_snipping(self):
        self.hide()  # Hide the main window
        self.snipping_widget = SnippingWidget()
        self.snipping_widget.show()


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == []:
        app = QApplication(sys.argv) 
        mainWindow = MainWindow()
        mainWindow.show()
        sys.exit(app.exec_())

    elif args[0] == 'train':
        if not os.path.exists(BUILD_DIR):
            print('Build directory does not exist. Creating one...')
            os.mkdir(BUILD_DIR)
        train_model()

    elif args[0] == 'test':
        if not os.path.exists(BUILD_DIR):
            raise Exception('Build directory does not exist. Please run \'make train\' first')
        test_model()
