import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *

if __name__ == '__main__':
    data = pd.read_csv('0.csv')
    # csv_writer = csv.writer(file)  # 储存全部数据
    print(data[:3])

    data.plot('Step', 'Value')
    plt.grid()
    plt.show()

