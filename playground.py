import matplotlib.pyplot as plt
import numpy as np

# накидываем тысячу точек от -3 до 3
x = np.linspace(-3, 3, 1000).reshape(-1, 1)

# задаём линейную функцию, которую попробуем приблизить нашей нейронной сетью
def f(x):    
    return x**2

f = np.vectorize(f)

# вычисляем вектор значений функции
y = f(x)

# создаём модель нейросети, используя Keras
from keras.models import Sequential
from keras.layers import Dense

def baseline_model():
    model = Sequential()
    model.add(Dense(2, input_dim=1, activation='sigmoid'))
    model.add(Dense(1, input_dim=2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

# тренируем сеть
model = baseline_model()
model.fit(x, y, nb_epoch=100, verbose = 0)

# отрисовываем результат приближения нейросетью поверх исходной функции
plt.scatter(x, y, color='black', antialiased=True)
plt.plot(x, model.predict(x), color='magenta', linewidth=2, antialiased=True)
plt.show()

# выводим веса на экран
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)
