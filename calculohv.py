import numpy as np
import matplotlib.pyplot as plt

x = [1.9901584082451123, 1.8630901353273945, 2.0179053768957544, 2.371420643583155, 2.3911580064093103,
     2.338721775365288, 2.325868004227871, 2.633196422209939, 2.782556421757145, 1.8936543890409197, 2.6760197433865165,
     2.118568111858813, 2.092659187462809, 1.9399332770231354, 2.0779914696115713, 2.112822553301288, 2.923949094956109,
     1.9901828839598301, 2.377845306951626]
y = [0.04226920305343192, 0.06058552201416078, 0.040901358309106524, 0.03610813503568599, 0.03587982706067336,
     0.0365649152126267, 0.036734692211283775, 0.033113373179836564, 0.03223268416828081, 0.052135004664888365,
     0.032429978352434975, 0.03950211120228833, 0.04028580486399307, 0.04630896560953162, 0.04058876500796342,
     0.039666123312507656, 0.030941544360590763, 0.04226666528209909, 0.0360742367628877]

x = np.sort(x)
y = np.sort(y)[::-1]

x_max = 3
y_max = 0.1

plt.figure(1)
yd=0
area=0

for i in range(len(x)):
    if i == 0:  # primer elemento
        plt.plot([x[i], x[i]], [y[i], y_max], 'g--',)  # vertical
        plt.plot([x_max, x[i]], [y_max, y_max], 'b--')  # horizontal
        print(i)
        x_d=x[i+1]-x[i]
        y_d=y_max-y[i]
        yd=y_d+(y[i]-y[i+1])
        print(yd)
        area=x_d*yd
        print(area)
    elif (0<i<len(x)-1):
        print(i)
        plt.plot([x[i-1], x[i-1]], [y[i], y[i - 1]], 'k--')  # vertical
        plt.plot([x[i - 1], x[i]], [y[i], y[i]], 'k--')  # horizontal
        x_d=x[i+1]-x[i]
        y_d=y[i]-y[i+1]
        print(yd)
        yd=y_d+yd
        print(yd)
        area=area+(yd*x_d)
        

    elif i == len(x)-1:  # ultimo elemento
        plt.plot([x[i-1], x[i-1]], [y[i], y[i - 1]], 'k--')  # vertical
        plt.plot([x[i - 1], x[i]], [y[i], y[i]], 'k--')  # horizontal
        plt.plot([x_max, x_max], [y[i], y_max], 'r--')  # vertical
        plt.plot([x[i], x_max], [y[i], y[i]], 'c--')  # horizontal
        x_d=x_max-x[i]
        area=area+(yd*x_d)
        

plt.plot(x, y, 'or')
plt.plot(x_max, y_max, 'ok')
print(area)
plt.show()
