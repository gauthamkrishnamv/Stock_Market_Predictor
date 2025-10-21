import datetime as dt
from time import sleep, time
import logging

import requests
import pandas as pd

from twelvedata import TDClient
from twelvedata.exceptions import TwelveDataError, BadRequestError

from collections import deque
from random import randint

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('miner_logs.log')
file_handler.setLevel(logging.DEBUG)
file_formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_formater = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formater)
m_logger.addHandler(file_handler)

TS = 0		#timestamp
COUNT = 1	#number of times called under a minute 

API_KEYS = {						    #TS, COUNT
	'7e7686bc76f248889426e1a8703b1b22': [0, 	0], #MINE
	'a9ce0f540604488c96abd7acd09eb33b': [0, 	0], #Vaishnav's
	'f2209d49a8cf4f77a9c7e1df4dd617c2': [0, 	0], #Vipin's
	'818ea2ce19374b21a21ffa1338c57e26': [0, 	0], #Gautham's
}


KEYS = deque(API_KEYS.keys())
run_out_of_credits = 'You have run out of API credits for the current minute.'

# KEYS.rotate(randint(1,5))
# td = None

def find_next_api():
	min_key = KEYS[0]
	min_time = 60

	for key, val in API_KEYS.items():
		if time() - val[TS] >= 60:
			KEYS.rotate(-KEYS.index(key))
			return 0

		
		else:
			rem = 60 - (time() - val[TS])
			if rem <= min_time:
				min_time = rem
				min_key = key
	
	KEYS.rotate( -KEYS.index(min_key))
	return abs(min_time)

def update_timestamp(apikey):
	ts, _ = API_KEYS[apikey]
	t = time()
	if t - ts >= 60:
		API_KEYS[apikey][TS] = t


def get_stocks_list(countries=['United States']):
	'''
		
		gets lists of supported stocks with their name and symbol
		and returns a list of strings 'name, symbol'

	'''

	url = 'https://api.twelvedata.com/stocks'
	stocks = []
	try:
		# params = {'exchange': 'NSE'}
		r = requests.get(url)

		for stock in r.json()['data']:
			if stock['country'] in countries and stock['type'] == 'Common Stock':

				name_symbol = f'{stock["name"]} | {stock["symbol"]}'
				stocks.append(name_symbol)

	except:
		pass

	return sorted(set(stocks), key=lambda s:s.split(',')[0])        


def next_start_end_date(df):
	end_date = df.index[0]
	delta = (df.index[-1] - end_date)
	start_date = end_date - dt.timedelta(days=delta.days)

	return start_date, end_date 


def get_total_rows(df_list):
	rows = 0
	for df in df_list:
		rows += len(df)
	return rows


def get_historical_price(symbol, data_points, interval, apikey=None):
	'''
	Returns the historical price data of the requested
	stock upto data_points number as a pandas data frame.

	'''
	
	timezone = "America/New_York"	
	df_list = []

	while True:
		td = TDClient(apikey=KEYS[0])
		if get_total_rows(df_list) >= data_points:
			break
		if df_list:
			end = df_list[0].index[0]
		else:
			end = None
		req_output_size = min(data_points - get_total_rows(df_list), 5000)
	    
		try:
			m_logger.debug(f'requesting {req_output_size}')
			df = td.time_series(
					symbol=symbol,
					interval=interval,
					outputsize=req_output_size,
					end_date=end,
					timezone=timezone,
					order='asc'
				).as_pandas()

			update_timestamp(KEYS[0])
			df_list.insert(0, df)
			m_logger.debug(f'got {len(df)}')

		except TwelveDataError as td:
				if run_out_of_credits in str(td):
					KEYS.rotate()
					# wait_time = find_next_api()
					m_logger.debug(f'sleeping for {1}')
					print(req_output_size, end='\r')
					sleep(1)
				
				else:
					data = pd.concat(df_list)
					return data

	data = pd.concat(df_list)
	return data[-data_points:]



if __name__ == '__main__':
	
	print(get_historical_price('AAPL', 10000, '1day'))
	# print(get_total_rows([]))
	# aapl = yf.Ticker('aapl')
	# historical_data = aapl.history(period='max', interval='1m', actions=False)
	# print(historical_data)
	# historical_data.to_pickle('apple.pkl')
	# print(yf.Ticker('aapl').history(period='max', interval='1d', actions=False))

	# print('setting time stamp')
	# for i, key in enumerate(API_KEYS):
	# 	update_timestamp(key)
	# 	print(i)
	# 	sleep(1)
	
	# wait_time = 100

	# while wait_time:
	# 	wait_time = find_next_api()
	# 	print(f'waiting for {wait_time} seconds')
	# 	sleep(wait_time)
	# 	KEYS.rotate()
	# 	print(f'after fake rotate {KEYS[0]}')
	# 	wait_time = find_next_api()
	# 	print(f'wt = {wait_time}, keys0= {KEYS[0]}')
		
