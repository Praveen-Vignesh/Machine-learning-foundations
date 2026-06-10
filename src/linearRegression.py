import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/pokemon.csv')

def loss_function(m, b, data):
    total_error = 0
    for i in range(len(data)): # summation of (y(i) - (mx(i) + b))^2 where i is from 1 to n
        x = data.iloc[i].hp
        y = data.iloc[i].attack
        total_error += (y - (m*x + b))**2
    
    return total_error / float(2*len(data))  # 1/2m * summation of (y(i) - (mx(i) + b))^2 where i is from 1 to n

def gradient_descent(m_now, b_now, data, rate):
    m_gradient, b_gradient = 0 , 0
    n = len(data)
    for i in range(n):
        x , y = data.iloc[i].hp, data.iloc[i].attack
        m_gradient += (1/n) * ((m_now*x + b_now) - y) * x
        b_gradient += (1/n) * ((m_now*x + b_now) - y)

    return m_now - m_gradient*rate, b_now - b_gradient*rate

m, b = 0, 0
rate = 0.0001
iterationsCount = 300

for i in range(iterationsCount):
    m, b = gradient_descent(m, b, data, rate)

print(m, b)

plt.scatter(data.hp, data.attack, color="green")
plt.plot(list(range(0, 260)), [m*x + b for x in range(0, 260)], color="red")
plt.show()