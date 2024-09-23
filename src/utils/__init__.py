import os
PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
BUILD_DIR = os.path.join(PROJECT_ROOT, 'build')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

STATE_DICT = 'im2latex_model.pth'
MODEL_INFO = 'im2latex_model_info.pth'
SCREENSHOT = 'screenshot.png'
