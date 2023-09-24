import tkinter as tk
from tkinter import filedialog
from predict import *
window = tk.Tk()
window.title('Eye disease predictor')
a=tk.StringVar()
b=tk.StringVar()
ent1=tk.Entry(window,textvariable=a)
ent1.place(x=200,y=50,width=500)
def browse_files():
    # Open a file dialog for browsing files
    filepath = filedialog.askopenfilename()

    # Display the selected file path in a label
    a.set(filepath)
#display final result
def predict_file():
    c=pred(a.get())

    if c[0, 0] == 1:
        lbl2=tk.Label(window,text="You are affected by CNV(Choroidal neovascularization).")
        lbl2.place(x=20,y=200)
    elif c[0, 1] == 1:
        lbl2=tk.Label(window,text="You are affected by DME(Diabetic Macular Edema).")
        lbl2.place(x=20, y=200)
    elif c[0, 2] == 1:
        lbl2=tk.Label(window,text="You are affected by DRUSEN.")
        lbl2.place(x=20, y=200)
    elif c[0, 3] == 1:
        lbl2=tk.Label(window,text="You eye's are perfectly alright.")
        lbl2.place(x=20, y=200)
    else:
        lbl2=tk.Label(window,text="Please enter correct path")
        lbl2.place(x=20, y=200)
    lbl3=tk.Label(window,text="If you are not able to read the above text,checkup with your eye docter😉",font=('arial',24))
    lbl3.place(x=20,y=300)

predict_button=tk.Button(window,text='predict',command=predict_file)
predict_button.place(x=400,y=100)
window.geometry('1000x1000')
lbl1=tk.Label(window,text="This window application is used to predict eye disease like(CNV,DME,DRUSEN,NORMAL)")
lbl1.place(x=0,y=20)
# Create a button for browsing files
browse_button = tk.Button(window, text="Browse Files", command=browse_files)
browse_button.place(x=200,y=100)
def cancle():
    a.set(' ')
clear_button=tk.Button(window,text="clear path",command=cancle)
clear_button.place(x=300,y=100)
# Create a label to display the selected file path
selected_file_label = tk.Label(window, text="Select your jpeg file path to predict eye disease:")
filepath=selected_file_label.place(x=50,y=50)
window.state('zoomed')
# Start the Tkinter event loop
window.mainloop()

