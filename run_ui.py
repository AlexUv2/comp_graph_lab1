import sys
import numpy as np
from typing import Dict

from PyQt5.QtWidgets import QMainWindow, QApplication, QShortcut
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsRectItem, QApplication, QGraphicsLineItem
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QPointF, QPoint, QLineF
from PyQt5 import QtCore, QtGui, QtWidgets

from config import axis_arr, ax_len, identity_matrix
from ui import Ui_MainWindow

POINTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
START_COORD = {'x': 2, 'y': 10}

LINES_LENGTHS: Dict[str, float] = {
    'AB': 50 / 10,
    'BC': 50 / 10,
    'CD': 10 / 10,
    'DE': 20 / 10,
    'EF': 150 / 10,
    'FG': 20 / 10,
    'GH': 55 / 10,
    'HI': 55 / 10,
    'IA': 20 / 10,
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()

        self.quitSc = QShortcut(QKeySequence(Qt.Key_Q), self)
        self.quitSc.activated.connect(QApplication.instance().quit)

        self.ui.setupUi(self)
        self.ui.graphicsView.scale(1, -1)

        self.ui.pushButton_length_apply.clicked.connect(self.group_box_length_value_changed)
        self.ui.pushButton_set_default.clicked.connect(self.set_default_figure)

        self.ui.spinBox_size.valueChanged.connect(self.size_value_changed)
        self.scene = QGraphicsScene(-10, -10, 2000, 2000)

        self.draw_coord_axis(matrix=identity_matrix)
        self.draw_figure(**self.get_default_points())
        self.ui.graphicsView.setScene(self.scene)

        print()

    def size_value_changed(self):
        self.points = self.get_current_points()
        self.scene.clear()
        self.draw_coord_axis(matrix=identity_matrix)
        self.draw_figure(**self.points)

    def group_box_length_value_changed(self):
        self.points = self.get_current_points()
        self.scene.clear()
        self.draw_coord_axis(matrix=identity_matrix)
        self.draw_figure(**self.points)

    def set_default_figure(self):
        self.points = self.get_default_points()

        for attr in vars(self):  # change values to default in spinboxes
            if 'spinBox_length' in attr:
                getattr(self, attr).setValue(int(attr.split('_')[-1].upper() * 10))

        self.scene.clear()
        self.draw_coord_axis(matrix=identity_matrix)
        self.draw_figure(**self.points)

    def draw_coord_axis(self, matrix: np.ndarray):
        unit_size = self.ui.spinBox_size.value()
        grid_num = 20

        O = axis_arr[0] @ matrix
        X = axis_arr[1] @ matrix
        Y = axis_arr[2] @ matrix

        O = np.divide(O, O[2])
        X = np.divide(X, X[2])
        Y = np.divide(Y, Y[2])

        O *= unit_size
        X *= unit_size
        Y *= unit_size

        O = O.astype(int)
        X = X.astype(int)
        Y = Y.astype(int)

        line_x = QLineF(QPoint(O[0], O[1]), QPoint(X[0], X[1]))
        line_y = QLineF(QPoint(O[0], O[1]), QPoint(Y[0], Y[1]))

        self.scene.addLine(line_x, QPen(Qt.red, 2, Qt.SolidLine))
        self.scene.addLine(line_y, QPen(Qt.green, 2, Qt.SolidLine))

        for i in range(1, grid_num):
            beg_x = np.array([0, i, 1]) @ matrix
            beg_x = np.divide(beg_x, beg_x[2])
            beg_x *= unit_size

            end_x = np.array([grid_num, i, 1]) @ matrix
            end_x = np.divide(end_x, end_x[2])
            end_x *= unit_size

            beg_y = np.array([i, 0, 1]) @ matrix
            beg_y = np.divide(beg_y, beg_y[2])
            beg_y *= unit_size

            end_y = np.array([i, grid_num, 1]) @ matrix
            end_y = np.divide(end_y, end_y[2])
            end_y *= unit_size

            beg_x = beg_x.astype(int)
            end_x = end_x.astype(int)
            beg_y = beg_y.astype(int)
            end_y = end_y.astype(int)

            line_x = QLineF(QPoint(beg_x[0], beg_x[1]), QPoint(end_x[0], end_x[1]))
            line_y = QLineF(QPoint(beg_y[0], beg_y[1]), QPoint(end_y[0], end_y[1]))
            self.scene.addLine(line_x, QPen(Qt.gray, 1, Qt.SolidLine))
            self.scene.addLine(line_y, QPen(Qt.gray, 1, Qt.SolidLine))

    def get_default_points(self) -> Dict[str, QPointF]:
        unit_size = self.ui.spinBox_size.value()
        point_a = QPointF(START_COORD['x'] * unit_size, START_COORD['y'] * unit_size)
        point_b = QPointF(point_a.x(), point_a.y() + LINES_LENGTHS['AB'] * unit_size)
        point_c = QPointF(point_b.x() + LINES_LENGTHS['BC'] * unit_size, point_b.y())
        point_d = QPointF(point_c.x(), point_c.y() + LINES_LENGTHS['CD'] * unit_size)
        point_e = QPointF(point_d.x() + LINES_LENGTHS['DE'] * unit_size, point_d.y())
        point_f = QPointF(point_e.x(), point_e.y() - LINES_LENGTHS['EF'] * unit_size)
        point_g = QPointF(point_f.x() - LINES_LENGTHS['FG'] * unit_size, point_f.y())
        point_h = QPointF(point_g.x(), point_g.y() + LINES_LENGTHS['GH'] * unit_size)
        point_i = QPointF(point_a.x() + LINES_LENGTHS['IA'] * unit_size, point_a.y())
        return {
            'point_a': point_a,
            'point_b': point_b,
            'point_c': point_c,
            'point_d': point_d,
            'point_e': point_e,
            'point_f': point_f,
            'point_g': point_g,
            'point_h': point_h,
            'point_i': point_i
        }

    def get_current_points(self) -> Dict[str, QPointF]:
        unit_size = self.ui.spinBox_size.value()

        point_a = QPointF(START_COORD['x'] * unit_size, START_COORD['y'] * unit_size)
        point_b = QPointF(point_a.x(), point_a.y() + self.ui.spinBox_length_ab.value() / 10 * unit_size)
        point_c = QPointF(point_b.x() + self.ui.spinBox_length_bc.value() / 10 * unit_size, point_b.y())
        point_d = QPointF(point_c.x(), point_c.y() + self.ui.spinBox_length_cd.value() / 10 * unit_size)
        point_e = QPointF(point_d.x() + self.ui.spinBox_length_de.value() / 10 * unit_size, point_d.y())
        point_f = QPointF(point_e.x(), point_e.y() - self.ui.spinBox_length_ef.value() / 10 * unit_size)
        point_g = QPointF(point_f.x() - self.ui.spinBox_length_fg.value() / 10 * unit_size, point_f.y())
        point_h = QPointF(point_g.x(), point_g.y() + self.ui.spinBox_length_gh.value() / 10 * unit_size)
        point_i = QPointF(point_a.x() + self.ui.spinBox_length_ia.value() / 10 * unit_size, point_a.y())
        return {
            'point_a': point_a,
            'point_b': point_b,
            'point_c': point_c,
            'point_d': point_d,
            'point_e': point_e,
            'point_f': point_f,
            'point_g': point_g,
            'point_h': point_h,
            'point_i': point_i
        }

    def draw_figure(self, **kwargs):

        line_ab = QLineF(kwargs['point_a'], kwargs['point_b'])
        line_bc = QLineF(kwargs['point_b'], kwargs['point_c'])
        line_cd = QLineF(kwargs['point_c'], kwargs['point_d'])
        line_de = QLineF(kwargs['point_d'], kwargs['point_e'])
        line_ef = QLineF(kwargs['point_e'], kwargs['point_f'])
        line_fg = QLineF(kwargs['point_f'], kwargs['point_g'])
        line_gh = QLineF(kwargs['point_g'], kwargs['point_h'])
        line_hi = QLineF(kwargs['point_h'], kwargs['point_i'])
        line_ia = QLineF(kwargs['point_i'], kwargs['point_a'])

        figure_pen = QPen(Qt.darkBlue, 2, Qt.SolidLine)

        self.scene.addLine(line_ab, figure_pen)
        self.scene.addLine(line_bc, figure_pen)
        self.scene.addLine(line_cd, figure_pen)
        self.scene.addLine(line_de, figure_pen)
        self.scene.addLine(line_ef, figure_pen)
        self.scene.addLine(line_fg, figure_pen)
        self.scene.addLine(line_gh, figure_pen)
        self.scene.addLine(line_hi, figure_pen)
        self.scene.addLine(line_ia, figure_pen)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


