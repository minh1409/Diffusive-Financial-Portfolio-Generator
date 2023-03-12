import numpy as np
from copy import deepcopy
from pprint import pprint
from scipy.optimize import linprog

class stock_environment:
  def read_data(self, file):
    return None

  def state(self, sub_stock, time):
    return None
  
  def reward(self, sub_stock, time, span, state, action, value_method="OPEN"):
    return None


class vnese_stock_environment(stock_environment):
  def __init__(self, file_path):
    super(vnese_stock_environment).__init__()
    self.data = dict()
    self.read_data(file_path)

  def read_data(self, file_path):
    with open(data) as stock_data:
      lines = stock_data.read().splitlines()
    for i in range(len(lines)):
      lines[i] = lines[i].split(sep=',')
    for i in range(len(lines)):
      temp = dict()
      temp["TIME"] = []
      temp["OPEN"] = []
      temp["HIGH"] = []
      temp["LOW"] = []
      temp["CLOSE"] = []
      temp["VOLUME"] = []
      temp["BUY TRANSACTION FEE"] = 0.0015
      temp["SELL TRANSACTION FEE"] = 0.0015
      self.data[lines[i][0]] = temp
    # 
    # STOCK_TICKER <=> 0
    # TIME <=> 1
    # OPEN <=> 2
    # HIGH <=> 3
    # LOW <=> 4
    # CLOSE <=> 5
    # VOLUME <=> 6
    #
    for i in range(len(lines)):
      if lines[i][0] != "<Ticker>":
        self.data[lines[i][0]]["TIME"].append(lines[i][1])
        self.data[lines[i][0]]["OPEN"].append(float(lines[i][2]))
        self.data[lines[i][0]]["HIGH"].append(float(lines[i][3]))
        self.data[lines[i][0]]["LOW"].append(float(lines[i][4]))
        self.data[lines[i][0]]["CLOSE"].append(float(lines[i][5]))
        self.data[lines[i][0]]["VOLUME"].append(float(lines[i][6]))
      temp_money = dict()
      temp_money["TIME"] = [1] * 3000
      temp_money["OPEN"] = [1] * 3000
      temp_money["HIGH"] = [1] * 3000
      temp_money["LOW"] = [1] * 3000
      temp_money["CLOSE"] = [1] * 3000
      temp_money["VOLUME"] = [1] * 3000
      temp_money["BUY TRANSACTION FEE"] = 0.0
      temp_money["SELL TRANSACTION FEE"] = 0.0
      self.data["MONEY"] = temp_money
  
def add_data_field(env, new_data, field_name):
  # DATA added in is in form of MAP STOCK_TICKER -> DATA
  for STOCK_TICKER in new_data.keys():
    env.data[STOCK_TICKER][field_name] = new_data[STOCK_TICKER]
  env.data["MONEY"][field_name] = None

def state(env, sub_stock, time, requested_field=["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]):
  ret = []
  for stock_iter in range(len(sub_stock)):
    temp = []
    for field in requested_field:
      temp.append(env.data[sub_stock[stock_iter]][field])
    ret.append(temp)
  return ret

def switch_distribution(holding, desired_distribution, price, fee_buy, fee_sell):
    # matrix buy_0 -> buy_n-1, sell_0 -> sell_n-1
    wealth = 0
    for i in range(len(price)):
        wealth += holding[i] * price[i]
    first_row = []
    for i in range(len(price)):
        first_row.append(1 / (1 - fee_buy[i]) * price[i])
    for i in range(len(price)):
        first_row.append((1 - fee_sell[i]) * price[i] * -1)
    matrix = [first_row]
    bias = [0]
    for i in range(len(price)):
        current = [0 for i in range(2 * len(price))]
        current[i] = price[i]
        current[len(price) + i] = price[i] * -1
        current_bias = price[i] * holding[i] * -1 + desired_distribution[i] * wealth
        for j in range(len(price)):
            current[j] -= desired_distribution[i] * price[j]
            current[len(price) + j] += desired_distribution[i] * price[j]
        matrix.append(deepcopy(current))
        bias.append(deepcopy(current_bias))
    c = [0 for i in range(2 * len(price))]
    for i in range(len(price)):
        c[i] = fee_buy[i] * price[i]
        c[len(price) + i] = fee_sell[i] * price[i]
    bound = [(0, None) for i in range(2 * len(price))]
    for i in range(len(price)):
        bound[len(price) + i] = (0, holding[i])
    res = linprog(c=c, A_eq=matrix, b_eq=bias, bounds=bound)
    return res['x']

def reward(env, sub_stock, time, span, state, action, value_method="OPEN"):
  constant = 1
  price = dict()
  ownership = dict()
  new_ownership = dict()
  for stock_iter in range(len(sub_stock)):
    price[sub_stock[stock_iter]] = env.data[sub_stock[stock_iter]][value_method][time:(time+span)]
    ownership[sub_stock[stock_iter]] = constant * state[stock_iter] / price[sub_stock[stock_iter]][0]
  inp_holding = [ownership[sub_stock[i]] for i in range(len(sub_stock))]
  inp_price = [price[sub_stock[i]][0] for i in range(len(sub_stock))]
  inp_fee_buy = [env.data[sub_stock[i]]["BUY TRANSACTION FEE"] for i in range(len(sub_stock))]
  inp_fee_sell = [env.data[sub_stock[i]]["SELL TRANSACTION FEE"] for i in range(len(sub_stock))]
  b_s_act = switch_distribution(holding=inp_holding, desired_distribution=action, price=inp_price, fee_buy=inp_fee_buy, fee_sell=inp_fee_sell)
  for stock_iter in range(len(sub_stock)):
    new_ownership[sub_stock[stock_iter]] = ownership[sub_stock[stock_iter]] + b_s_act[stock_iter] - b_s_act[len(sub_stock) + stock_iter]
  c_1 = 0
  c_2 = 0
  for stock_iter in range(len(sub_stock)):
    c_1 += price[sub_stock[stock_iter]][-1] * ownership[sub_stock[stock_iter]]
    c_2 += price[sub_stock[stock_iter]][-1] * new_ownership[sub_stock[stock_iter]]
  return {"profit":c_2 - c_1}