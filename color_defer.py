import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QFileDialog, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QHBoxLayout,
                             QScrollArea, QMenuBar, QMenu)
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QBrush,
                         QIcon, QCursor)  # Добавлен QCursor
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal


class ImageLabel(QLabel):
    # Сигнал для уведомления о завершении выделения
    selection_finished = pyqtSignal()
    # Сигнал для обновления метки зума
    zoom_changed = pyqtSignal(int)
    # Сигнал для запроса прокрутки (панорамирования)
    pan_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False

        # Параметры панорамирования
        self.is_panning = False
        self.pan_start = QPoint()

        # Параметры масштабирования
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_step = 0.1
        self.original_pixmap = None

    def wheelEvent(self, event):
        """Обработка прокрутки колёсика мыши для масштабирования"""
        # Если зажата средняя кнопка (панорамирование), игнорируем зум колесом
        # Хотя обычно wheelEvent срабатывает на скролл, а не на клик,
        # но на всякий случай проверяем, не панорамируем ли мы (хотя это разные события)
        if not self.original_pixmap:
            return

        # Определяем направление прокрутки
        delta = event.angleDelta().y()

        if delta > 0:
            # Прокрутка вверх - увеличение
            new_zoom = self.zoom_level + self.zoom_step
        else:
            # Прокрутка вниз - уменьшение
            new_zoom = self.zoom_level - self.zoom_step

        # Ограничиваем уровень масштабирования
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        # Применяем масштабирование только если уровень изменился
        if abs(new_zoom - self.zoom_level) > 0.001:
            self.zoom_level = new_zoom
            self.update_pixmap()
            # Испускаем сигнал об изменении зума
            self.zoom_changed.emit(int(self.zoom_level * 100))

    def update_pixmap(self):
        """Обновление отображаемого pixmap с учётом масштабирования"""
        if self.original_pixmap:
            new_width = int(self.original_pixmap.width() * self.zoom_level)
            new_height = int(self.original_pixmap.height() * self.zoom_level)
            scaled_pixmap = self.original_pixmap.scaled(
                new_width, new_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self.setFixedSize(scaled_pixmap.size())

            # Меняем курсор на "руку", если изображение больше области просмотра
            # (Это эвристика, точный размер области просмотра здесь неизвестен,
            # но визуально понятно, что изображение можно двигать)
            self.setCursor(Qt.ArrowCursor)

    def set_original_pixmap(self, pixmap):
        """Установка оригинального pixmap и инициализация масштабирования"""
        self.original_pixmap = pixmap
        self.zoom_level = 1.0
        self.update_pixmap()

    def get_scaled_coordinates(self, point):
        """Преобразование координат с учётом масштабирования"""
        if self.zoom_level > 0:
            x = int(point.x() / self.zoom_level)
            y = int(point.y() / self.zoom_level)
            return QPoint(x, y)
        return point

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            # Начало панорамирования
            self.is_panning = True
            # ИСПОЛЬЗУЕМ ГЛОБАЛЬНЫЕ КООРДИНАТЫ (экран)
            self.pan_start = event.globalPos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            self.selection_start = self.get_scaled_coordinates(event.pos())
            self.selection_end = self.selection_start
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_panning:
            # Вычисляем смещение на основе глобальных координат
            # Это гарантирует стабильность, даже если виджет двигается
            delta = event.globalPos() - self.pan_start

            # Испускаем сигнал для прокрутки ScrollArea
            self.pan_requested.emit(-delta.x(), -delta.y())

            # Обновляем начальную точку для следующего шага
            # Важно: обновляем именно глобальную позицию
            self.pan_start = event.globalPos()
            return

        if self.is_selecting:
            self.selection_end = self.get_scaled_coordinates(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.update()
            # Испускаем сигнал о завершении выделения
            self.selection_finished.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.is_selecting and self.selection_start and self.selection_end:
            painter = QPainter(self)
            # Тонкая обводка
            #painter.setPen(QPen(QColor(255, 0, 0, 180), 1, Qt.SolidLine))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0, 50))

            # Масштабируем координаты выделения для отображения
            start_scaled = QPoint(
                int(self.selection_start.x() * self.zoom_level),
                int(self.selection_start.y() * self.zoom_level)
            )
            end_scaled = QPoint(
                int(self.selection_end.x() * self.zoom_level),
                int(self.selection_end.y() * self.zoom_level)
            )

            rect = QRect(start_scaled, end_scaled).normalized()
            painter.drawRect(rect)

    def reset_zoom(self):
        """Сброс масштабирования к оригинальному размеру"""
        self.zoom_level = 1.0
        self.update_pixmap()

    def get_zoom_level(self):
        """Получение текущего уровня масштабирования в процентах"""
        return int(self.zoom_level * 100)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Average Color Finder")
        self.setGeometry(100, 100, 1200, 800)

        self.image = None
        self.pixmap = None
        self.current_color = None

        # Создаем центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Создаём ScrollArea для прокрутки при большом масштабе
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        # Включаем полосы прокрутки (по умолчанию включены, но явно укажем политику)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Контейнер для image_label внутри scroll_area
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container.setLayout(self.container_layout)

        self.image_label = ImageLabel(self.container)
        self.container_layout.addWidget(self.image_label)

        self.scroll_area.setWidget(self.container)

        # Подключаем сигналы к слотам
        self.image_label.selection_finished.connect(self.calculate_average_color)
        self.image_label.zoom_changed.connect(self.update_zoom_label)
        # Подключаем сигнал панорамирования
        self.image_label.pan_requested.connect(self.handle_pan)

        # Метка для отображения уровня масштабирования
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("font-weight: bold; padding: 5px;")

        # Кнопки управления масштабированием
        self.zoom_in_button = QPushButton("Zoom In (+)")
        self.zoom_in_button.clicked.connect(lambda: self.adjust_zoom(0.1))

        self.zoom_out_button = QPushButton("Zoom Out (-)")
        self.zoom_out_button.clicked.connect(lambda: self.adjust_zoom(-0.1))

        self.zoom_reset_button = QPushButton("Reset Zoom (100%)")
        self.zoom_reset_button.clicked.connect(self.reset_zoom)

        # Создаем таблицу для хранения цветов
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["№", "R", "G", "B", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(200)

        # Кнопка для добавления цвета в таблицу
        self.add_button = QPushButton("Add Color to Table")
        self.add_button.clicked.connect(self.add_color_to_table)
        self.add_button.setEnabled(False)

        # Кнопка для открытия изображения
        self.open_button = QPushButton("📂 Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setStyleSheet("font-size: 12px; padding: 8px;")

        # Layout для кнопок масштабирования
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_reset_button)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_label)

        # Layout для основных кнопок
        main_buttons_layout = QHBoxLayout()
        main_buttons_layout.addWidget(self.open_button)
        main_buttons_layout.addWidget(self.add_button)

        # Основной layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(main_buttons_layout)
        main_layout.addLayout(zoom_layout)
        main_layout.addWidget(self.scroll_area)
        main_layout.addWidget(self.table)

        self.central_widget.setLayout(main_layout)

        # Создаем меню
        self.create_menu()

        # Инструкция
        self.info_label = QLabel("📌 Scroll wheel to zoom | Middle click drag to pan | Left click to select area")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        main_layout.addWidget(self.info_label)

    def create_menu(self):
        """Создание меню приложения"""
        menubar = self.menuBar()

        # Меню File
        file_menu = menubar.addMenu('&File')

        open_action = file_menu.addAction('&Open Image\tCtrl+O')
        open_action.triggered.connect(self.open_image)

        file_menu.addSeparator()

        exit_action = file_menu.addAction('E&xit\tCtrl+Q')
        exit_action.triggered.connect(self.close)

        # Меню Zoom
        zoom_menu = menubar.addMenu('&Zoom')

        zoom_in_action = zoom_menu.addAction('Zoom &In\tCtrl++')
        zoom_in_action.triggered.connect(lambda: self.adjust_zoom(0.1))

        zoom_out_action = zoom_menu.addAction('Zoom &Out\tCtrl+-')
        zoom_out_action.triggered.connect(lambda: self.adjust_zoom(-0.1))

        zoom_reset_action = zoom_menu.addAction('&Reset Zoom\tCtrl+0')
        zoom_reset_action.triggered.connect(self.reset_zoom)

        # Меню Help
        help_menu = menubar.addMenu('&Help')

        about_action = help_menu.addAction('&About')
        about_action.triggered.connect(self.show_about)

    def handle_pan(self, dx, dy):
        """Обработка запроса на панорамирование от ImageLabel"""
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()

        # Обновляем значения скроллбаров
        h_bar.setValue(h_bar.value() + dx)
        v_bar.setValue(v_bar.value() + dy)

    def adjust_zoom(self, delta):
        """Изменение уровня масштабирования через кнопки"""
        new_zoom = self.image_label.zoom_level + delta
        new_zoom = max(self.image_label.min_zoom, min(self.image_label.max_zoom, new_zoom))
        self.image_label.zoom_level = new_zoom
        self.image_label.update_pixmap()
        self.update_zoom_label()

    def reset_zoom(self):
        """Сброс масштабирования"""
        self.image_label.reset_zoom()
        self.update_zoom_label()

    def update_zoom_label(self, zoom_percent=None):
        """Обновление отображения уровня масштабирования"""
        if zoom_percent is None:
            zoom_percent = self.image_label.get_zoom_level()
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")

    def open_image(self):
        """Открытие изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, image_path):
        """Загрузка изображения"""
        self.image = QImage()
        if not self.image.load(image_path):
            print(f"Failed to load image: {image_path}")
            return False

        self.pixmap = QPixmap.fromImage(self.image)
        self.image_label.set_original_pixmap(self.pixmap)
        self.update_zoom_label()
        self.setWindowTitle(f"Average Color Finder - {image_path}")
        return True

    def calculate_average_color(self):
        """Вычисление среднего цвета в выделенной области"""
        if (not hasattr(self.image_label, 'selection_start') or
                not self.image_label.selection_start or
                not self.image_label.selection_end or
                not self.image):
            return

        # Координаты уже масштабированы в ImageLabel
        rect = QRect(self.image_label.selection_start,
                     self.image_label.selection_end).normalized()

        # Проверяем, что область выделения не нулевая
        if rect.width() == 0 or rect.height() == 0:
            return

        ptr = self.image.bits()
        ptr.setsize(self.image.byteCount())

        # Определяем формат изображения
        if self.image.format() == QImage.Format_RGB32 or self.image.format() == QImage.Format_ARGB32:
            arr = np.array(ptr).reshape(self.image.height(), self.image.width(), 4)
        elif self.image.format() == QImage.Format_RGB888:
            arr = np.array(ptr).reshape(self.image.height(), self.image.width(), 3)
            # Добавляем альфа-канал для совместимости
            arr = np.dstack([arr, np.full((self.image.height(), self.image.width()), 255, dtype=np.uint8)])
        else:
            # Конвертируем в RGB32 для совместимости
            self.image = self.image.convertToFormat(QImage.Format_RGB32)
            ptr = self.image.bits()
            ptr.setsize(self.image.byteCount())
            arr = np.array(ptr).reshape(self.image.height(), self.image.width(), 4)

        x1, y1 = rect.topLeft().x(), rect.topLeft().y()
        x2, y2 = rect.bottomRight().x(), rect.bottomRight().y()

        x1 = max(0, min(x1, self.image.width() - 1))
        y1 = max(0, min(y1, self.image.height() - 1))
        x2 = max(0, min(x2, self.image.width() - 1))
        y2 = max(0, min(y2, self.image.height() - 1))

        region = arr[y1:y2 + 1, x1:x2 + 1]
        avg_color = np.mean(region[:, :, :3], axis=(0, 1))
        avg_color = np.round(avg_color).astype(int)

        self.current_color = QColor(avg_color[2], avg_color[1], avg_color[0])
        self.add_button.setEnabled(True)

        print(f"Average color in selection: RGB({avg_color[2]}, {avg_color[1]}, {avg_color[0]})")
        print(f"Hex: {self.current_color.name()}")
        print(f"Zoom level: {self.image_label.get_zoom_level()}%")

    def add_color_to_table(self):
        """Добавление цвета в таблицу"""
        if not self.current_color:
            return

        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        # Номер
        self.table.setItem(row_position, 0, QTableWidgetItem(str(row_position + 1)))

        # R, G, B компоненты
        self.table.setItem(row_position, 1, QTableWidgetItem(str(self.current_color.red())))
        self.table.setItem(row_position, 2, QTableWidgetItem(str(self.current_color.green())))
        self.table.setItem(row_position, 3, QTableWidgetItem(str(self.current_color.blue())))

        # Квадратик с цветом
        color_item = QTableWidgetItem()
        color_item.setBackground(QBrush(self.current_color))
        color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row_position, 4, color_item)

        # Делаем кнопку неактивной после добавления
        self.add_button.setEnabled(False)

    def show_about(self):
        """Показ информации о программе"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About Average Color Finder",
            "<h2>Average Color Finder</h2>"
            "<p>Version 1.1</p>"
            "<p>A tool for finding average colors in image regions.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Zoom with mouse wheel</li>"
            "<li>Pan with Middle Mouse Button drag</li>"
            "<li>Select region with left mouse button</li>"
            "<li>Calculate average color</li>"
            "<li>Save colors to table</li>"
            "</ul>"
        )

    def keyPressEvent(self, event):
        """Обработка нажатий клавиш"""
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.adjust_zoom(0.1)
            elif event.key() == Qt.Key_Minus:
                self.adjust_zoom(-0.1)
            elif event.key() == Qt.Key_0:
                self.reset_zoom()
            elif event.key() == Qt.Key_O:
                self.open_image()
            elif event.key() == Qt.Key_Q:
                self.close()
        super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())