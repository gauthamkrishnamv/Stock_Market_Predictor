import os
import sys
import pickle
import logging

from collections import OrderedDict as odict
from time import sleep

from PyQt5.QtWidgets import *
from PyQt5 import uic, Qt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator
# from PyQt5.QtChart import QChart, QChartView, QLineSeries

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pandas as pd
import numpy as np
import joblib

from brains import create_model, load_model, predict, scale_and_split_data, join
from console import Console, ProcessOutputReader

from miner import ( 
	# get_stocks_list, 
	get_historical_price, 
	# API_KEYS, 
	KEYS, 
	TwelveDataError,
	BadRequestError
)

# sys.setrecursionlimit(2000)

MAX_INT = 2147483647

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log_v1_file.log')
file_handler.setLevel(logging.DEBUG)
file_formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_formater = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formater)
logger.addHandler(file_handler)

user = 'user'
trained_stocks_file = os.path.join(user, 'TRAINED_STOCKS.txt')
untrained_stocks_file = os.path.join(user, 'UNTRAINED_STOCKS.txt')
# models = os.path.join(user, 'models')

intervals_buttons = {
	'1min': 'min_1', 
	'5min': 'min_5', 
	'15min': 'min_15', 
	'30min': 'min_30', 
	'45min': 'min_45', 
	'1h': 'hr_1', 
	'2h': 'hr_2',
	'4h': 'hr_4', 
	'1day': 'day_1',
	'1week': 'week_1'
}

intervals_pd_map = {
	'1min': '1min',
	'5min': '5min',
	'15min': '15min',
	'30min': '30min',
	'45min': '45min',
	'1h': '1H',
	'2h': '2H',
	'4h': '4H',
	'1day': '1D',
	'1week': '1W'
}


#pages
TRAINING, TRAINED, PREDICT = range(3)


def first_run():
	pass

def val(obj):
	return int(obj.text())

def models_path(file_name: str):
	if not file_name:
		return	'user\\models'

	return os.path.join('user', 'models', file_name)

class MainWindow(QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()
		uic.loadUi('main_window.ui', self)
		self.setWindowTitle('Stock Predictor')
		
		self.canvas = None
		self.selected_stock: QListWidgetItem = None
		self.selected_interval = '1day'

		self.parameter_string = ''
		self.current_model_params = odict()  #updated in evt_combo_box_index_changed if all cb contains text or in add_items_to_combo_boxes if self.temp_model

		self.temp_model:bool = False
		self.current_page = TRAINING
		self.adding_items_to_cb = False

		self.graph_stacked_widget = self.findChild(QStackedWidget, 'graph_stacked_widget')
		self.right_stacked_widget = self.findChild(QStackedWidget, 'right_stacked_widget')
		self.status_bar = self.findChild(QStatusBar, 'status_bar')

		self.use_button = self.findChild(QPushButton, 'use_button')
		self.use_button.clicked.connect(self.goto_predict_page)

		self.create_new_model_button = self.findChild(QPushButton, 'create_new_model_button')
		self.create_new_model_button.clicked.connect(self.evt_create_new_model_button_clicked)

		self.use_another_model_button = self.findChild(QPushButton, 'use_another_model_button')
		self.use_another_model_button.clicked.connect(self.goto_trained_model_page)
		
		self.stock_data_graph = self.findChild(QWidget, 'stock_data_widget')
		self.stock_data_graph.setLayout(QVBoxLayout())

		self.goto_training_page()
		# self.goto_trained_model_page()
		# self.goto_predict_page()
		self.show()

	def selected_stock_symbol(self):
		return self.selected_stock.text().split('|')[1].strip()

	@staticmethod
	def combo_box_remove_items(combo_box):
		for i in range(combo_box.count()):
			combo_box.removeItem(i)

	def create_parameter_string(self, *args):
		''' Used to create the self.parameter_string while training a model'''
		self.parameter_string = ''
		if args:
			for parameter in args:
				self.parameter_string += str(parameter) + '_'
			self.parameter_string = self.parameter_string[:-1]
			return

		self.parameter_string = self.selected_stock_symbol()
		self.parameter_string += '_' + self.selected_interval
		self.parameter_string += '_' + self.no_of_data_points_input.text()
		for line_edit in self.model_settings_input:
			self.parameter_string += '_' + \
			self.model_settings_input[line_edit].text()
		
		logger.debug(f'paramter_string = {self.parameter_string} from create_parameter_string')


	def goto_trained_model_page(self):
		if self.current_page == PREDICT:
			self.predictor_thread.stop()

		if self.current_page != TRAINED:
			self.create_graph_canvas(xlabel='Epochs', ylabel='Accuracy')
		
		self.current_page = TRAINED
		self.right_stacked_widget.setCurrentIndex(1)
		self.init_filter_model()
		# self.update_result_graph()
		if self.temp_model:
			self.evt_reset_stock_list()
			self.update_result_graph()


	def goto_predict_page(self):
		if self.temp_model:
			# convert to temp model to trained and set the parameter string
			self.trained_stocks(self.selected_stock.text())
			logger.debug(f'parameter string: {self.parameter_string} from goto_predict_page')
			model = load_model(join('model.h5'))
			model.save(models_path(self.parameter_string + '.h5'))
			del model

			scaler = joblib.load(join('scaler.sclr'))
			joblib.dump(scaler, models_path(self.parameter_string + '.sclr'))
			del scaler

			with open(join('history'), 'rb') as hf:
				history = pickle.load(hf)

			with open(models_path(self.parameter_string), 'wb') as hf:
				pickle.dump(history, hf)

			del history


		else:
			# set parameter string using combo boxes
			if not self.current_model_params:
				self.show_dialog("Insufficiant data", "Fill all combo boxes!!")
				return
				
		self.temp_model = False
		self.current_page = PREDICT
		self.update_model_details_label()
		self.create_graph_canvas()
		self.update_prediction_graph()
		# self.graph_stacked_widget.setCurrentIndex(2)
		self.right_stacked_widget.setCurrentIndex(2)

	def goto_training_page(self):
		if self.current_page == TRAINED:
			pass		
		self.current_page = TRAINING
		self.create_graph_canvas()
		self.temp_model = False
		self.graph_stacked_widget.setCurrentIndex(0)
		self.right_stacked_widget.setCurrentIndex(0)
		self.list_stocks()
		self.init_model_stock_config()


	def evt_create_new_model_button_clicked(self):
		self.goto_training_page()

	
	def remove_widgets_of_layout(self, layout):
		for i in reversed(range(layout.count())): 
			layout.itemAt(i).widget().setParent(None)

	def evt_status_signal(self, message):
		self.status_bar.showMessage(message)

	def show_dialog(self, title, message):
		msg = QMessageBox()
		msg.setIcon(QMessageBox.Information)
		msg.setText(message)
		msg.setWindowTitle(title)
		msg.setStandardButtons(QMessageBox.Ok)
		msg.exec_()


	def trained_stocks(self, stock=None):
		trained = []
		if stock:
			with open(trained_stocks_file, 'a+') as file:
					file.write(f'{stock}\n')

		with open(trained_stocks_file, 'r+') as file:
			trained = [name.strip() for name in file.readlines()]
		
		if self.temp_model and self.selected_stock.text() not in trained:
			trained.append(self.selected_stock.text())

		trained.sort()
		return trained

	def untrained_stocks(self, stocks=None):
		untrained = []
		
		with open(untrained_stocks_file, 'r+') as file:
			if stocks:
				print(*stocks, sep='\n', file=file)

			untrained = [name.strip() for name in file.readlines()]
		
		trained = self.trained_stocks()
		untrained = [item for item in untrained if item not in trained]

		untrained.sort()
		return untrained
	

	def init_model_stock_config(self):

		# initializing stock settings input
		def evt_interval_button_clicked(interval):
			if not self.selected_stock:
				return

			self.selected_interval = interval
			self.update_stock_graph()
		
		stock_config_frame = self.right_stacked_widget.findChild(QFrame, 'stock_config_frame')
		
		for interval, button_name in intervals_buttons.items():
			button = stock_config_frame.findChild(QPushButton, button_name)
			button.clicked.connect(
				lambda state, interval=interval: evt_interval_button_clicked(interval)
			)

		self.no_of_data_points_input = stock_config_frame.findChild(
			QLineEdit, 
			'no_of_data_points_input'	
		)
		self.no_of_data_points_input.setValidator(QIntValidator(1000, 15000))
		# initializing model settings input
		model_settings_frame = self.right_stacked_widget.findChild(QFrame, 'model_settings_input_frame')
		self.input_names_and_validation = {
			'input_size_input': (2, 1000), 
			'output_size_input': (1, 50), 
			'epochs_input': (1, 200), 
			'batch_size_input': (1, 100), 
			'no_of_dense_layers_input': (1, 50), 
			'no_of_dense_neurons_input': (5, 100), 
			'no_of_lstm_layers_input': (2, 50), 
			'no_of_lstm_neurons_input': (5, 100)
		}

		self.model_settings_input = odict()

		for name, limits in self.input_names_and_validation.items():
			line_edit = model_settings_frame.findChild(QLineEdit, name)
			line_edit.setValidator(
				QIntValidator(*limits)
			)
			line_edit.setPlaceholderText(f'{limits[0]} - {limits[1]}')
			self.model_settings_input[name] = line_edit


		self.train_button = self.findChild(QPushButton, 'train_button')
		self.train_button.clicked.connect(self.train_model)

	def init_filter_model(self):			
		self.reset_filters_button = self.right_stacked_widget.findChild(QPushButton, 'reset_filters_button')
		self.reset_filters_button.clicked.connect(
			lambda state: self.add_items_to_combo_boxes(reset=True)
		)
		
		self.combo_box_names = [
			'time_frame_cb',
			'no_of_data_points_cb',
			'input_size_cb',
			'output_size_cb',
			'epochs_cb',
			'batch_size_cb',
			'no_of_dense_layers_cb',
			'no_of_dense_neurons_cb',
			'no_of_lstm_layers_cb',
			'no_of_lstm_neurons_cb'
		]
		
		self.combo_boxes = odict()
		for name in self.combo_box_names:
			combo_box = self.right_stacked_widget.findChild(QComboBox, name)
			combo_box.currentIndexChanged.connect(
				lambda state, name=name: self.evt_combo_box_index_changed(name)
			)
			self.combo_boxes[name] = combo_box

		self.add_items_to_combo_boxes()


	def evt_combo_box_index_changed(self, name=None):
		if self.adding_items_to_cb or self.current_page != TRAINED:
			return
		
		# logger.debug(f'{name} combo box index changed')
		current_params = odict()
		current_params['stock_symbol'] = self.selected_stock_symbol()
		# current_params['stock_symbol'] = 'ABVM'

		for name, cb in self.combo_boxes.items():
			if cb.currentText():
				current_params[name] = cb.currentText()
			else:
				current_params[name] = 0

		model_params = self.get_all_model_params()
		matched_models = []
		# logger.debug(f'len model params: {len(model_params)}, current_params: {current_params}')
		for i, parameter in enumerate(current_params):
			parameter_val = current_params[parameter]
			if parameter_val:
				matched_models = []
				for model in model_params:
					if model[i] == parameter_val:
						matched_models.append(model)
			
				model_params = matched_models
		# model_params = matched_models
		logger.debug(f'matched models after cb index changed: {(model_params)}')
		self.add_items_to_combo_boxes(model_params=model_params)


		for name in self.combo_box_names:
			if not self.combo_boxes[name].currentText():
				return
			
		self.current_model_params = current_params
		# logger.debug(f'current model parameters from cb index changed: {self.current_model_params}')
		self.update_result_graph()
			

	def add_items_to_combo_boxes(self, model_params=None, reset=False):
		self.adding_items_to_cb = True
		if model_params == None:
			model_params = self.get_all_model_params()

		cb: QComboBox = None
		for j, name in enumerate(self.combo_box_names):
			cb = self.combo_boxes[name]
			current_text = '' if reset else cb.currentText()
			cb.clear()
			
			if not self.temp_model:
				if current_text:
					cb.addItem(current_text)
					# logger.debug(f'current text of {name} = {current_text}')
					continue

				cb_items = set()
				for model in model_params:
					# logger.debug(f"length of model: {len(model)}, model: {model}")
					# logger.debug()
					cb_items.add(str(model[j+1]))
					# cb.addItem(str(model[j+1]))
				cb.addItems(sorted(list(cb_items)))
				cb.setCurrentIndex(-1)
				# logger.debug(f'cb_items = {cb_items}')
			
			
		if self.temp_model:
			params = self.parameter_string.split('_')
			self.current_model_params['stock_symbol'] = params[0]
			params = params[1:]
			for param, cb_name in zip(params, self.combo_boxes):
				self.combo_boxes[cb_name].addItem(str(param)) 
				self.combo_boxes[cb_name].setCurrentIndex(0)
				self.current_model_params[cb_name] = param

		self.adding_items_to_cb = False

	@staticmethod
	def get_all_model_params():
		''''Returns a split up parameters all saved models'''
		model_params = os.listdir(models_path(''))
		model_params = list(
			filter(
				lambda f:os.path.splitext(f)[1] not in ['.h5', '.sclr'], 
				model_params
			)
		)
		for i, p in enumerate(model_params):
			model_params[i] = p.split('_')

		return model_params			

			
	def train_model(self):
		def evt_finished_training(exit_code):
			logger.debug(f'Model trained with exit_code = {exit_code}.')
			if exit_code < 0:
				self.remove_widgets_of_layout(
					self.stock_data_graph.layout()
				)
				self.create_graph_canvas()
				self.temp_model = False
				return
			
			self.temp_model = True
			self.status_bar.showMessage('Model Trained.')
			self.goto_trained_model_page()

		def error_handler(e):
			if e == 0:
				self.status_bar.show_message('Failed to download Data. Network Error!')

		def not_valid(line_edit):
			btm = line_edit.validator().bottom()
			top = line_edit.validator().top()
			valid = line_edit.validator().validate(
				line_edit.text(),
				0
			)[0]

			if valid != 2:
				self.status_bar.showMessage(
					f'''{" ".join(line_edit.objectName().split('_')[:-1])} 
					should be between {btm} and {top}'''
				)
				return True

			return False

		self.status_bar.showMessage('')

		if not_valid(self.no_of_data_points_input) or not self.selected_stock:
			return

		for line_edit in self.model_settings_input:
			if not_valid(self.model_settings_input[line_edit]):
				return

		self.create_parameter_string()

		reader = ProcessOutputReader()
		console = Console(reader)
		reader.produce_output.connect(console.setPlainText)
		reader.finished.connect(evt_finished_training)

		layout = self.stock_data_graph.layout()
		self.remove_widgets_of_layout(layout)

		layout.addWidget(console)
		layout.update()
		
		
		self.status_bar.showMessage('Creating Model...')
		self.creator = ModelerThread(self)
		self.creator.status_signal.connect(self.evt_status_signal)
		self.creator.start()
		self.creator.wait()
		self.creator.stop()
		del self.creator

		self.status_bar.showMessage('Training... do not interact with the ui, Press Ctrl + C to kill the training.')
		epochs = self.model_settings_input['epochs_input'].text()
		batches = self.model_settings_input['batch_size_input'].text()
		self.disable_widgets()
		reader.start('python', ['-u', 'brains.py', epochs, batches])
		# reader.start('python', ['-u', 'test_process.py', epochs, batches])			
		self.temp_model = True
	
	def disable_widgets(self, exceptions=None): #exceptions are the widgets to be left alone
		pass

	
	def list_stocks(self):
		def evt_update_stocks_list(stocks):
			self.untrained_stocks(stocks)
			self.untrained_list.addItems(stocks)
			self.status_bar.showMessage("Downloading Complete")
			self.lister_thread.stop()
			del self.lister_thread

			self.init_search_bar() #search bar function

		def evt_stock_list_row_changed(item, name):
			if self.temp_model:
				return

			if item:
				self.selected_stock = item
			
			if self.current_page == TRAINING:
				if name == 'T':
					logger.debug("Going to trained model page")
					self.goto_trained_model_page()
				
				else:
					self.update_stock_graph()
			
			elif self.current_page == TRAINED:
				self.evt_combo_box_index_changed()

		def evt_trained_stock_list_row_changed(item, name):
			if self.temp_model:
				return
			
			if item:
				self.selected_stock = item
			
			if self.current_page == TRAINING:
				logger.debug('Going to trained model page')
				# layout = self.stock_data_graph.layout()
				# self.remove_widgets_of_layout(layout)
				self.goto_trained_model_page()
				return
			
			if self.current_page == TRAINED:
				self.add_items_to_combo_boxes(reset=True)
				self.evt_combo_box_index_changed()
			
		def evt_untrained_stock_list_row_changed(item, name):
			if self.temp_model:
				return

			if self.current_page == TRAINED:
				return
			
			if item:
				self.selected_stock = item

			
		self.trained_list: QListWidget = self.findChild(QListWidget, 'trained_list')
		self.trained_list.currentRowChanged.connect(
			lambda state, name='T': evt_trained_stock_list_row_changed(self.trained_list.currentItem(), name)
		)
		self.trained_list.clear()
		self.trained_list.addItems(self.trained_stocks())

		self.untrained_list: QListWidget = self.findChild(QListWidget, 'untrained_list')
		self.untrained_list.currentRowChanged.connect(
			lambda state, name='U' :evt_untrained_stock_list_row_changed(self.untrained_list.currentItem(), name)
		)
		self.untrained_list.clear()

		self.status_bar.showMessage("Downloading List of Stocks")
		self.lister_thread = ListerThread(self)
		self.lister_thread.start()
		self.lister_thread.update_untrained_list.connect(evt_update_stocks_list)
		

	def evt_reset_stock_list(self):
			self.trained_list.clear()
			self.trained_list.addItems(self.trained_stocks())
			
			self.untrained_list.clear()
			self.untrained_list.addItems(self.untrained_stocks())


	def init_search_bar(self):
		def evt_update_stock_list():
			inp_text = self.search_bar.text()
			if inp_text == '':
				self.evt_reset_stock_list()
				return

			matched_items = [
				i.text() for i in self.trained_list.findItems(
					inp_text, Qt.Qt.MatchContains
				)
			]
			if matched_items:
				self.trained_list.clear()
				self.trained_list.addItems(matched_items)

			matched_items = [i.text() for i in self.untrained_list.findItems(inp_text, Qt.Qt.MatchContains)]
			if matched_items:
				self.untrained_list.clear()
				self.untrained_list.addItems(matched_items)

		self.search_bar = self.findChild(QLineEdit, 'search_bar')
		self.search_bar.textChanged.connect(evt_update_stock_list)
		# self.search_bar.returnPressed.connect(self.evt_reset_stock_list)

	
	def update_model_details_label(self):
		details = ""
		for key, value in self.current_model_params.items():
			param_name = key.split('_')[:-1]
			param_name = " ".join(param_name)
			param_name = param_name.title()
			details += f'{param_name}: {value}\n'
		
		model_details_label = self.right_stacked_widget.findChild(QLabel, 'model_details_label')
		model_details_label.setText(details.strip())


	def create_graph_canvas(self, nrows=1, ncols=1, xlabel='Time', ylabel='price'):
		plt.style.use('dark_background')
		figure = plt.figure(figsize=(15,6), layout='tight')
		self.canvas = FigureCanvas(figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		layout = self.stock_data_graph.layout()
		self.remove_widgets_of_layout(layout)
		
		layout.addWidget(self.toolbar)
		layout.addWidget(self.canvas)

		self.ax = self.canvas.figure.subplots(
			nrows=nrows,
			ncols=ncols,
			subplot_kw={
				# 'autoscale_on': True,
				'autoscaley_on': True,
				'xlabel': xlabel,
				'ylabel': ylabel,
				'frame_on': False
			}
		)
		# self.ax.autoscale()
		self.plot = None
		self.canvas.draw()
		logger.debug(f'Creating graph canvas current page = {self.current_page}')

	def update_stock_graph(self, item=None, interval='1day'):

		def evt_update_graph(stock_data):
			logger.debug(f'plot: {self.plot}, type(plot): {type(self.plot)}, stock_data: {len(stock_data)}')

			if self.plot:
				try:
					self.plot[0].remove()		
				except ValueError:
					pass

			if not stock_data.empty:
				self.ax.set_title(f'{self.selected_stock.text()} at {interval} interval')
				self.plot = self.ax.plot(stock_data['close'])
				self.ax.set_xlim(stock_data.index[0], stock_data.index[-1])
				self.ax.set_ylim(*self.price_lim(stock_data))
				self.canvas.draw()
				self.status_bar.showMessage('Dowloaded ' + self.stock_details)

			else:
				self.status_bar.showMessage('Stock Data unavailable. Choose another stock')

			self.worker_thread.stop()
			del self.worker_thread
		

		if item:
			self.selected_stock = item


		if self.selected_interval:
			interval = self.selected_interval

		stock_name, symbol = self.selected_stock.text().split('|')
		symbol = symbol.strip()
		stock_name = stock_name.strip()
		
		self.stock_details = f'{symbol} at {interval} interval.'
		status = 'Downloading price data of '
		self.status_bar.showMessage(status + self.stock_details)

		self.worker_thread = PriceDownloader(symbol, interval)
		self.worker_thread.start()
		self.worker_thread.finished.connect(evt_update_graph)


	def update_result_graph(self):	#temp = True means the results displayed are of temporary model
		def limits(data):
			mn = min(data)
			mx = max(data)
			rng = mx - mn
			pcnt = rng * 0.05
			return mn - pcnt, mx + pcnt

		logger.debug(f'Executing update result graph with temp={self.temp_model}')		
		if self.temp_model:
			with open(join('history'), 'rb') as hf:
				history = pickle.load(hf)
			logger.debug(f'history = {history}')

		else:
			parameters = []
			for value in self.current_model_params.values():
				parameters.append(value)

			history = self.get_model_history(parameters)
		
		if len(history['accuracy']) == 1:
			history['accuracy'].insert(0, 0)

		if self.plot:
			try:
				self.plot[0].remove()		
			except ValueError:
				pass
		
		self.ax.set_title('Epoch Accuracy Graph')
		self.plot = self.ax.plot(history['accuracy'])
		self.ax.set_xlim(0, len(history['accuracy'])+1)
		self.ax.set_ylim((limits(history['accuracy'])))
		self.canvas.draw()
		self.status_bar.showMessage('Updated result graph.')
	
	def update_prediction_graph(self):	#temp = True means the results displayed are of temporary model
		def price_lim(data):
			mn = data['close'].min()
			mx = data['close'].max()
			
			if data['prediction'][0] == 0:
				mn = min(mn, data['prediction'].min())
				mx = max(mx, data['prediction'].max())
				print(f' data["prediction"][0] = { data["prediction"][0]}')

			rng = mx - mn
			pcnt = rng * 0.2
			return mn - pcnt, mx + pcnt
		
		def evt_update_prediction_graph(prediction_data):
			logger.debug(f'Ploting prediction data close: {prediction_data}')
			if self.plot:
				try:
					for plot in self.plot:
						plot.remove()		
				except ValueError:
					pass

			if not prediction_data.empty:
				self.ax.set_title(f'Prediction graph')
				# print(f'prediction data = {prediction_data}')
				self.plot = self.ax.plot(prediction_data['close'])
				if prediction_data['prediction'][-1] != 0:
					self.plot = self.ax.plot(prediction_data['prediction'])
					self.ax.legend(['Closing Price', 'Predicted Price'], loc='lower right')
				print(f'plot = {self.plot} ')
				self.ax.set_xlim(prediction_data.index[0], prediction_data.index[-1])
				self.ax.set_ylim(*price_lim(prediction_data))
				self.canvas.draw()
				# self.status_bar.showMessage('Dowloaded ' + self.stock_details)

			else:
				self.status_bar.showMessage('Prediction Data unavailable.')


		logger.debug(f'Executing update prediction graph')

		self.status_bar.showMessage('Predicting...')
		self.predictor_thread = PredictorThread(self)
		self.predictor_thread.start()
		self.predictor_thread.predicted.connect(evt_update_prediction_graph)
		self.predictor_thread.status_signal.connect(self.evt_status_signal)

		
	@staticmethod
	def price_lim(stock_data):
		mn = stock_data['close'].min()
		mx = stock_data['close'].max()
		rng = mx - mn
		pcnt = rng * 0.05
		return mn - pcnt, mx + pcnt


	def get_model_history(self, parameters=None):
		if parameters:
			self.create_parameter_string(*parameters)

		# self.parameter_string = 'AMZN_1day_10000_50_2_1_3_2_25_2_50'
		with open(models_path(self.parameter_string), 'rb') as hf:
			history = pickle.load(hf)

		return history

	def get_model(self, parameters=None):
		if parameters:
			self.create_parameter_string(*parameters)
		name = self.parameter_string + '.h5'
		
		return load_model(models_path(name))

	def get_scaler(self, parameters=None):
		if parameters:
			self.create_parameter_string(*parameters)
		sclr_name = self.parameter_string + '.sclr'
		
		return joblib.load(models_path(sclr_name))
	

class ListerThread(QThread):
	update_untrained_list = pyqtSignal(list)

	def __init__(self, caller_obj):
		super().__init__()
		self.caller_obj = caller_obj

	def run(self):
		logger.info('Lister Thread Started.')
		# stocks = get_stocks_list()
		stocks = self.caller_obj.untrained_stocks()
		# stocks = ['Agilent Technologies, Inc. | A', 'Alcoa Corp | AA', 'Aareal Bank AG | AAALF', 'Asia Broadband, Inc. | AABB', 'Aberdeen International Inc. | AABVF', 'AAC Holdings Inc | AAC', 'AAC Technologies Holdings Inc | AACAF', 'Armada Acquisition Corp. I | AACI', 'Armada Acquisition Corp. I Unit | AACIU', 'Armada Acquisition Corp. I | AACIW']
		self.update_untrained_list.emit(stocks)	

	def stop(self):
		self.is_running = False
		self.terminate()
		logger.info('Lister Thread Stopped.')


class PriceDownloader(QThread):
	finished = pyqtSignal(pd.DataFrame)
	status_signal = pyqtSignal(str)
	# finished_for_training = pyqtSignal(pd.DataFrame, int)

	def __init__(self, symbol, interval):
		super().__init__()
		self.symbol = symbol
		self.interval = interval
		# logger.debug(f'Called Downloader thread for training = {for_training}')

	def run(self):
		logger.info('Price Downloader Thread Started.')
		# self.status_signal.emit('Downloading Prce Data')
		stock_data = pd.DataFrame()
		
		try:
			stock_data = get_historical_price(self.symbol, 1000, self.interval)
		
		except BadRequestError:
			self.status_signal.emit('Requested bad stock data!')
			return
		
		self.finished.emit(stock_data)
			
		
	def stop(self):
		self.is_running = False
		self.terminate()
		logger.info('Price Downloader Thread Stopped.')


class ModelerThread(QThread):
	error_signal = pyqtSignal(int)
	status_signal = pyqtSignal(str)

	def __init__(self, caller:MainWindow):
		super().__init__()
		self.c = caller
		# self.symbol = self.c.selected_stock.text().split('|')[1]
		# self.interval = self.c.selected_interval
		self.symbol = 'AAPL'
		self.interval = '1day'
		# logger.debug(f'from modeler thread temp stock data: {self.c.temp_stock_data}')

	def run(self):
		logger.info("Modeler Thread Started")
		self.status_signal.emit('Downloading stock data...')
		
		stock_data = pd.DataFrame()
		try:
			stock_data = get_historical_price(
				self.symbol, 
				10000, #val(self.c.no_of_data_points_input), 
				self.interval, 
			)
		
		except BadRequestError:
			self.status_signal.emit('Requested bad stock data!')
			self.error_signal.emit(0)
			return

		# if len(stock_data) < val(self.c.no_of_data_points_input):
		# 	self.status_signal.emit('Insufficiant data downloaded')
		# 	return

		self.status_signal.emit('Formating Data...')
		train, test, scaler = scale_and_split_data(
			stock_data, 
			val(self.c.model_settings_input['input_size_input']),
			val(self.c.model_settings_input['output_size_input'])
		)
		logger.debug(f'stock data\'s shape = {stock_data.shape}')
		# train, test, scaler = scale_and_split_data( stock_data, 50, 2)

		np.save( join('x_train.npy'), train[0], allow_pickle=False)
		np.save( join('y_train.npy'), train[1], allow_pickle=False)
		np.save( join('x_test.npy'), test[0], allow_pickle=False)
		np.save( join('y_test.npy'), test[1], allow_pickle=False)
		joblib.dump(scaler, join('scaler.sclr'))

		self.c.status_bar.showMessage('Creating Model!')
		# model = create_model( 50, 2, 2, 50, 2, 25 )
		model = create_model(
			val(self.c.model_settings_input['input_size_input']),
			val(self.c.model_settings_input['output_size_input']),
			val(self.c.model_settings_input['no_of_lstm_layers_input']),
			val(self.c.model_settings_input['no_of_lstm_neurons_input']),
			val(self.c.model_settings_input['no_of_dense_layers_input']),
			val(self.c.model_settings_input['no_of_dense_neurons_input']),
		)
		model.save(join('model.h5'))
		self.status_signal.emit('Model compiled.')

		logger.debug('Completed execution of Modeler Thread :)')
	
	def stop(self):
		self.is_running = False
		self.terminate()
		logger.info("Modeler Thread Stopped.")


class PredictorThread(QThread):
	predicted = pyqtSignal(pd.DataFrame)
	status_signal = pyqtSignal(str)
	error_signal = pyqtSignal(int)

	def __init__(self, caller:MainWindow):
		super().__init__()

		self.c = caller
		# self.c.current_model_params = odict([('stock_symbol', 'AMZN'), ('time_frame_cb', '1day'), ('no_of_data_points_cb', '10000'), ('input_size_cb', '50'), ('output_size_cb', '2'), ('epochs_cb', '1'), ('batch_size_cb', '3'), ('no_of_dense_layers_cb', '2'), ('no_of_dense_neurons_cb', '25'), ('no_of_lstm_layers_cb', '2'), ('no_of_lstm_neurons_cb', '50')])
		# self.c.create_parameter_string(*self.c.current_model_params.values())

	def run(self):
		logger.info("Predictor Thread Started")
		logger.debug(f'From Predictor Thread with current model parameters: {self.c.current_model_params}')
		while True:
			stock_data = pd.DataFrame()
			try:
				stock_data = get_historical_price(
					symbol=self.c.current_model_params['stock_symbol'],
					data_points=int(self.c.current_model_params['input_size_cb']),
					interval=self.c.current_model_params['time_frame_cb']
				)
			
			except BadRequestError:
				self.status_signal.emit('Requested bad stock data!')
				self.error_signal.emit(0)
				return
			
			stock_data['prediction'] = [0] * len(stock_data)
			self.predicted.emit(stock_data)
			self.status_signal.emit('Loading model Data')

			model = self.c.get_model()
			scaler = self.c.get_scaler()
			interval_pd_format = intervals_pd_map[self.c.selected_interval]
			data = predict(model, scaler, stock_data, interval_pd_format)
			self.predicted.emit(data)

			self.status_signal.emit('')
			sleep(60)


	def stop(self):
		self.is_running = False
		self.terminate()
		logger.info("Predictor Thread Stopped")


if __name__ == '__main__':
	first_run()

	app = QApplication(sys.argv)

	window = MainWindow()
	
	sys.exit(app.exec_())
