from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import pickle
import pandas as pd


win = Tk()
win.title('START UP PROFIT PREDICTION SYSTEM')
win.config(bg='lightblue')
win.geometry('800x600')


with open('encoder', 'rb') as f:
    oh = pickle.load(f)


with open('scaler', 'rb') as f:
    sc = pickle.load(f)


with open('model', 'rb') as f:
    reg = pickle.load(f)


def predict():
    rd = float(r1.get())
    ad = float(a1.get())
    mar = float(m1.get())
    state = s1.get()
    d1 = {'R&D Spend': [rd], 'Administration':
          [ad], 'Marketing Spend': [mar], 'State': [state]}

    newdf = pd.DataFrame(d1)
    newdf[oh.get_feature_names_out()] = oh.transform(newdf[['State']])
    newdf.drop(columns=oh.feature_names_in_, inplace=True)
    newdf[['R&D Spend', 'Administration', 'Marketing Spend']] = sc.transform(newdf[['R&D Spend',
                                                                                   'Administration', 'Marketing Spend']])
    result = reg.predict(newdf)
    final = result[0]
    mystr = f'Predicted profit is {final:.2f} $'
    l5.config(text=mystr)


l0 = Label(win, text='START UP PROFIT PREDICTION SYSTEM', width=40, bg='white',
           fg='black', bd=5, relief='ridge', font=('times new roman', 18, 'bold'))


l0.place(x=268, y=20)


# RD spend   admin spend marketing spend state


# RD spend and entry


l1 = Label(win, text='R&D Spend', width=20, bg='white',
           fg='black', bd=5, relief='ridge', font=('times new roman', 12, 'bold'))


l1.place(x=100, y=80)


r1 = StringVar()
e1 = Entry(win, width=40, bg='white', textvariable=r1,
           fg='black', bd=5, relief='ridge', font=('times new roman', 12, 'bold'))


e1.place(x=310, y=80)


# #admin >> label ,entry


l2 = Label(win, text='Administration', width=20, bg='white',
           fg='black', bd=5, relief='ridge', font=('times new roman', 12, 'bold'))


l2.place(x=100, y=130)


a1 = StringVar()
e2 = Entry(win, width=40, bg='white', textvariable=a1,
           fg='black', bd=5, relief='ridge', font=('times new roman', 12, 'bold'))


e2.place(x=310, y=130)


# marketing >> label entry


l3 = Label(win, text='Marketing Spend', bg='white', fg='black', width=20, bd=5,
           relief='ridge', font=('times new roman', 12, 'bold'))
l3.place(x=100, y=180)


m1 = StringVar()
e3 = Entry(win, textvariable=m1, bg='white', fg='black', width=40, bd=5,
           relief='ridge', font=('times new roman', 12, 'bold'))
e3.place(x=310, y=180)


# state >> label entry


l4 = Label(win, text='State', bg='white', fg='black', width=20, bd=5,
           relief='ridge', font=('times new roman', 12, 'bold'))
l4.place(x=100, y=230)


s1 = StringVar()
e4 = ttk.Combobox(win, textvariable=s1, width=38,
                  font=('times new roman', 12, 'bold'))
e4.place(x=310, y=230, height=30)
e4['values'] = ('California', 'Florida', 'New York')
e4.current(0)


# Button


b1 = Button(win, text='PREDICT', width=12, command=predict, bd=5, relief='ridge',
            bg='black', fg='white', font=('times new roman', 11, 'bold'))
b1.place(x=250, y=300)


# extra label


l5 = Label(win, text='PROFIT PREDITION IS HERE !!!', bg='white', fg='black', width=40, height=5, bd=5,
           relief='ridge', font=('times new roman', 16, 'bold'))
l5.place(x=700, y=230)


win.mainloop()
