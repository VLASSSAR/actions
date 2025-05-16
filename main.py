import numpy as np
import random

#Гиперпараметры
INPUT_DIM = 4 #Размер входного вектора, так как 4 признака
OUT_DIM = 3 #Три класса ириса (SETOSA, VERSICOLOR, VIRGINICA)
H_DIM = 10 #Количество нейронов в первом слое

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True) #Суммируем по измерению 1, измерение 0 - не трогаем

def sparse_cross_entropy(z, y):
    return -np.log(z[0, y]) #По-хорошему, нам надо было здесь написать сумму, но так как в полном векторе правильного ответа у нас везде нули и лишь одна единица, от суммы останется лишь одна компонента вектора Z, как раз компонента с индексом Y(индекс правильного класса)

def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))])) #Посчитаем кроссэнтропию для каждого элемента батча, то есть для каждого вектора-строки из Z и для соответствующего правильного индекса из Y.


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes)) #Вектор-строка из нулей
    y_full[0, y] = 1 #В индекс y присваиваем 1
    return y_full #Возвращаем вектор

def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full
def relu_deriv(t):
    return (t >= 0).astype(float)


#x = np.random.randn(1, INPUT_DIM) #Вектор с признаками Ириса
#y = random.randint(0, OUT_DIM-1) #Правильный ответ (вектор истинного распределения)

from sklearn import datasets
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
print(dataset)

W1 = np.random.rand(INPUT_DIM, H_DIM) #Матрица W1
b1 = np.random.rand(1, H_DIM) #Вектор-строка смещения
W2 = np.random.rand(H_DIM, OUT_DIM) #Матрица W2
b2 = np.random.rand(1, OUT_DIM) #Вектор-строка смещения

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)

ALPHA = 0.0002
NUM_EPOCHS = 400
BATCH_SIZE = 50

loss_arr = [] #Список

for ep in range(NUM_EPOCHS):
    random.shuffle(dataset) #Чтобы примеры показывались нейросети всегда в новом порядке
    for i in range(len(dataset) // BATCH_SIZE):

        batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE]) #Чтобы привести к матрице X и вектору Y проведем такую манипуляцию
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Forward (Прямое распространение)
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y)) #От Z-наших вероятностей и правильных ответов Y (называется sparse, потому что Y-не вектор распределения как в стандартном выражении для CE, а индекс правильного класса.


        # Backward (Обратное распространение ошибки)
        y_full = to_full_batch(y, OUT_DIM) #Сначала получим полный вектор правильного ответа, функция to_full будет превращать индекс правильного класса в соответствующее распределение (вектор из нулей и единицы) - One-Hot Encoding
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        #Update (Обновление весов)
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        #Сделаем метрики для отслеживания ошибки
        loss_arr.append(E)

def predict(x): #X - вектор с признаками
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2) #Z - вектор из вероятностей
    return z
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z) #Выбираем индекс класса, который нам предсказывает нейросеть с помощью argmax.
        if y_pred == y: #Проверяем, совпадает ли он с истинным индексом Y.
            correct += 1
    acc = correct / len(dataset)
    return acc

accuracy = calc_accuracy()
print("Accuracy:", accuracy)

#Нарисуем график падения по иттерациям, воспользуемся matplotlib
import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()