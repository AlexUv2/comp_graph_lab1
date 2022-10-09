import sys
import string
import numpy as np
from typing import Dict, List

from PyQt5.QtWidgets import QMainWindow, QShortcut
from PyQt5.QtWidgets import QGraphicsScene, QApplication
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QPointF, QPoint, QLineF

from config import axis_arr, identity_matrix
from ui import Ui_MainWindow


POINTS = string.ascii_lowercase[:9]
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
        self.ui.pushButton_affine_apply.clicked.connect(self.apply_affine_transform)
        self.ui.pushButton_perspective_apply.clicked.connect(self.apply_perspective_transform)
        self.ui.pushButton_rotation_apply.clicked.connect(self.apply_rotation)

        self.ui.spinBox_size.valueChanged.connect(self.size_value_changed)
        self.scene = QGraphicsScene(-10, -10, 2000, 2000)


        self.set_default_figure()
        self.points_transform_matrics = identity_matrix

        self.ui.graphicsView.setScene(self.scene)

        print()

    def size_value_changed(self):
        self.points = self.get_points_multiplied_on_matrix(self.points_transform_matrics, is_perspective=self.is_perpective)
        self.scene.clear()
        self.draw_coord_axis(matrix=self.axis_transform_matrix)
        self.draw_figure(**self.points)

    def group_box_length_value_changed(self):
        self.points = self.get_points_multiplied_on_matrix(self.points_transform_matrics, is_perspective=self.is_perpective)
        self.scene.clear()
        self.draw_coord_axis(matrix=self.axis_transform_matrix)
        self.draw_figure(**self.points)

    def apply_affine_transform(self):
        self.is_perpective = False
        self.points_transform_matrics = identity_matrix

        self.axis_transform_matrix = self.get_affine_transform_matrix()
        self.points_transform_matrics = self.get_affine_transform_matrix()
        self.points = self.get_points_multiplied_on_matrix(self.points_transform_matrics, is_perspective=self.is_perpective)
        self.scene.clear()
        self.draw_coord_axis(matrix=self.axis_transform_matrix)
        self.draw_figure(**self.points)

    def apply_perspective_transform(self):
        self.is_perpective = True
        self.points_transform_matrics = identity_matrix

        self.axis_transform_matrix = self.get_perspective_transform_matrix()
        self.points_transform_matrics = self.get_perspective_transform_matrix()
        self.points = self.get_points_multiplied_on_matrix(self.points_transform_matrics, is_perspective=self.is_perpective)
        self.scene.clear()
        self.draw_coord_axis(matrix=self.axis_transform_matrix)
        self.draw_figure(**self.points)

    def apply_rotation(self):
        unit_size = self.ui.spinBox_size.value()

        # self.is_perpective = False
        self.points_transform_matrics = self.get_rotation_matrix()
        self.points_transform_matrics = self.get_rotation_matrix()
        self.points = self.get_points_multiplied_on_matrix(
            self.points_transform_matrics,
            is_perspective=self.is_perpective,
            is_rotation=True
        )

        self.scene.clear()
        x = self.ui.spinBox_rotation_x.value()  # * unit_size
        y = self.ui.spinBox_rotation_y.value()  # * unit_size

        # draw blue point on rotation point
        x, y, mult = np.array([x, y, 1]) @ self.axis_transform_matrix
        if self.is_perpective:
            x /= mult
            y /= mult

        x *= unit_size
        y *= unit_size
        self.scene.addEllipse(x - 2, y - 2, 4, 4, QPen(Qt.blue), QBrush(Qt.blue))


        self.draw_coord_axis(matrix=self.axis_transform_matrix)
        self.draw_figure(**self.points)


    def set_default_figure(self):
        self.points = self.get_default_points()
        self.is_perpective = False

        for attr in vars(self.ui):  # change values to default in spinboxes
            if 'spinBox_length' in attr:
                getattr(self.ui, attr).setValue(int(LINES_LENGTHS[attr.split('_')[-1].upper()] * 10))

        self.axis_transform_matrix = identity_matrix
        self.points_transform_matrics = identity_matrix
        self.scene.clear()
        self.draw_coord_axis(matrix=self.axis_transform_matrix)
        self.draw_figure(**self.points)

    def get_affine_transform_matrix(self) -> np.ndarray:
        xx = self.ui.spinBox_affine_xx.value()
        xy = self.ui.spinBox_affine_xy.value()

        yx = self.ui.spinBox_affine_yx.value()
        yy = self.ui.spinBox_affine_yy.value()

        ox = self.ui.spinBox_affine_0x.value()
        oy = self.ui.spinBox_affine_0y.value()

        return np.array(
            [[xx, xy, 0],
             [yx, yy, 0],
             [ox, oy, 1]]
        )

    def get_perspective_transform_matrix(self) -> np.ndarray:
        xx = self.ui.spinBox_perspective_xx.value()
        xy = self.ui.spinBox_perspective_xy.value()
        wx = self.ui.spinBox_perspective_wx.value()

        yx = self.ui.spinBox_perspective_yx.value()
        yy = self.ui.spinBox_perspective_yy.value()
        wy = self.ui.spinBox_perspective_wy.value()

        ox = self.ui.spinBox_perspective_0x.value()
        oy = self.ui.spinBox_perspective_0y.value()
        wo = self.ui.spinBox_perspective_w0.value()

        return np.array(
            [[xx * wx, xy * wx, wx],
             [yx * wy, yy * wy, wy],
             [ox * wo, oy * wo, wo]]
        )

    def get_rotation_matrix(self) -> np.ndarray:
        unit_size = self.ui.spinBox_size.value()

        x = self.ui.spinBox_rotation_x.value() #* unit_size
        y = self.ui.spinBox_rotation_y.value() #* unit_size
        x, y, mult = np.array([x, y, 1]) @ self.axis_transform_matrix

        if self.is_perpective:
            x /= mult
            y /= mult
        else:
            x *= unit_size
            y *= unit_size

        angle = self.ui.spinBox_rotation_angle.value()
        sin_a = np.sin(np.radians(angle))
        cos_a = np.cos(np.radians(angle))

        return np.array(
            [[cos_a,    sin_a,   0],
             [-sin_a,   cos_a,   0],
             [-x*(cos_a-1) + (y*sin_a), -x*sin_a-y*(cos_a-1), 1]]
        )

    def get_points_multiplied_on_matrix(
            self,
            transform_matrix: np.ndarray,
            is_perspective: bool = False,
            is_rotation: bool = False
    ) -> Dict[str, QPointF]:
        unit_size = self.ui.spinBox_size.value()
        cur_points = self.get_current_points()

        if is_rotation:
            cur_points = self.points

        if is_perspective:
            points_matrix: List[np.ndarray] = [np.array([point.x() / unit_size, point.y() / unit_size, 1]) for point in cur_points.values()]
        else:
            points_matrix: List[np.ndarray] = [np.array([point.x(), point.y(), 1]) for point in cur_points.values()]

        points_transformed = [point @ transform_matrix for point in points_matrix]

        if is_perspective:
            points_transformed = [(point / point[-1]) * unit_size for point in points_transformed]

        return {f'point_{point_index}': QPointF(*point[:2]) for point, point_index in zip(points_transformed, POINTS)}

    def draw_coord_axis(self, matrix: np.ndarray):
        unit_size = self.ui.spinBox_size.value()
        grid_num = 200  ###### ADD SPINBOX

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
