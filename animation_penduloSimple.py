#!/usr/bin/env python
import PySimpleGUI as sg
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
"""
Demonstrates one way of embedding Matplotlib figures into a PySimpleGUI window.

Paste your Pyplot code into the section marked below.

Do all of your plotting as you normally would, but do NOT call plt.show(). 
Stop just short of calling plt.show() and let the GUI do the rest.

The remainder of the program will convert your plot and display it in the GUI.
If you want to change the GUI, make changes to the GUI portion marked below.

"""


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

#fig = plt.gcf()
# fig2= plt.gcf()

#plt.rcParams['animation.html'] = 'html5'
#TODO ESTO ES DE LAYOUT
#figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
layout = [[sg.Text('Plot test', font='Any 18')],
          [sg.Canvas( key='-CANVAS-')],
          [sg.OK( size=(4, 2))]]

window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI',
                   layout,finalize=True)



# ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE -------------------------------
def limcontro(u):
    if (u >= 0):
        if (u > 2.94):
            ur = 2.94
        elif (u <= 2.94):
            ur = u

    else:

        if (u >= -2.94):
            ur = u
        else:
            ur = -2.94

    return ur


'''Time parameters'''
dt = 0.005  # Tiempo de muestreo (5ms)
ti = 0.0  # Tiempo inicial de la simulación (0s)
tf = 10.0  # Tiempo inicial de la simulación (10s)
n = int((tf - ti) / dt) + 1  # Número de muestras
t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)

'''Dynamic parameters'''
m = 0.5  # Masa del pendulo (kg)
l = 1.0  # Longitud de la barra del péndulo (m)
lc = 0.3  # Longitud al centro de masa del péndulo (m)
b = 0.05  # Coeficiente de fricción viscosa pendulo
g = 9.81  # Aceleración de la gravedad en la Tierra
I = 0.006  # Tensor de inercia del péndulo

'''State variables'''
x = np.zeros((n, 2))

'''Control vector'''
u = np.zeros((n, 1))

'''Initial conditions'''
x[0, 0] = 0  # Initial pendulum position (rad)
x[0, 1] = 0  # Initial pendulum velocity (rad/s)
ie_th = 0
ise = 0
ise_next = 0
iadu = 0
iadu_next = 0

'''State equation'''
xdot = [0, 0]

'''Dynamic simulation'''
for i in range(n - 1):
    '''Current states'''

    th = x[i, 0]
    th_dot = x[i, 1]

    '''Controller'''
    e_th = np.pi - th
    e_th_dot = 0 - th_dot

    Kp = 9.00809903857079  # 1.57302981266663
    Kd = 0.74331509706173  # 0.292240643773894
    Ki = 0

    u[i] = limcontro(Kp * e_th + Kd * e_th_dot + Ki * ie_th)

    '''System dynamics'''
    xdot[0] = th_dot
    xdot[1] = (u[i] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)

    '''Integrate dynamics'''
    x[i + 1, 0] = x[i, 0] + xdot[0] * dt
    x[i + 1, 1] = x[i, 1] + xdot[1] * dt
    ie_th = ie_th + e_th * dt
    ise = ise_next + (e_th ** 2) * dt
    iadu = iadu_next + (abs(u[i] - u[i - 1])) * dt

    ise_next = ise
    iadu_next = iadu

u[n - 1] = u[n - 2]

print(x[:, 0])
print(ise)
print(iadu)

'''Plotting results'''
#plt.figure(figsize=(6, 5))
# plt.subplot(221)
# plt.plot(t, x[:, 0], 'k', lw=1)
# plt.legend([r'$\theta$'], loc=1)
# plt.ylabel('Pendulum position')
# plt.xlabel('Time')
#
# plt.subplot(222)
# plt.plot(t, x[:, 1], 'b', lw=1)
# plt.legend([r'$\dot{\theta}$'], loc=1)
# plt.ylabel('Pendulum speed')
# plt.xlabel('Time')
#
# plt.subplot(223)
# plt.plot(t, u[:, 0], 'r', lw=2)
# plt.legend([r'$u$'], loc=1)
# plt.ylabel('Control signal')
# plt.xlabel('Time')

#plt.show()

'''Animation'''



x0 = np.zeros(len(t))
y0 = np.zeros(len(t))

x1 = l * np.sin(x[:, 0])
y1 = -l * np.cos(x[:, 0])

fig = plt.figure(figsize=(6, 5.2))
ax = fig.add_subplot(111, autoscale_on=False,xlim=(-1.8, 1.8), ylim=(-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('y')
fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

line, = ax.plot([], [], 'o-', color='orange', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
time_template = 't= %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    line.set_data([x0[i], x1[i]], [y0[i], y1[i]])
    time_text.set_text(time_template % t[i])
    return line, time_text,

ani_a = animation.FuncAnimation(fig, animate, \
                                np.arange(1, len(t)), \
                                interval=40, blit=False)
#plt.show()
#fig_agg.draw()

# ------------------------------- END OF YOUR MATPLOTLIB CODE -------------------------------

# ------------------------------- Beginning of Matplotlib helper code -----------------------

# ------------------------------- Beginning of GUI CODE -------------------------------


#fig_photo = draw_figure(window['-CANVAS-'].TKCanvas, fig)
#fig_photo = draw_figure(window['-CANVASAN-'].TKCanvas,fig)

event, values = window.read()
#event, values = window.read()
window.close()