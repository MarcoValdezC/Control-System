import PySimpleGUI as sg
import time

layout = [      
    [sg.Canvas(size=(100, 100), background_color='red', key= 'canvas')],      
    [sg.T('Change circle color to:'), sg.Button('Red'), sg.Button('Blue')]      
    ]

window = sg.Window('Canvas test')      
window.Layout(layout)      
window.Finalize()

canvas = window.FindElement('canvas')      
cir = canvas.TKCanvas.create_oval(50, 50, 100, 100)

while True:      
    event, values = window.Read()
    '''
    if event is None:      
        break      
    if event == 'Blue':      
        canvas.TKCanvas.itemconfig(cir, fill="Blue")      
    elif event == 'Red':      
        canvas.TKCanvas.itemconfig(cir, fill="Red")

    '''

# this is the part that I need to sort out

    for i in range(10):
        if i % 2 == 0:
            canvas.TKCanvas.itemconfig(cir, fill="Blue")

        else:
            canvas.TKCanvas.itemconfig(cir, fill="Red")
