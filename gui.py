from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import json
import numpy as np
from src.Network import *
# from misc import sigmoid
import tensorflow as tf

data=load("model/model.txt")#load data from file
weights = [np.array(w) for w in data["weights"]]
biases = [np.array(b) for b in data["biases"]]

def predict_digit(img):
	#resize image to 28x28 pixels
    img = img.resize((28,28))
    
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)

#to set image array such that it can matchs to our input data
    img=255-img
    for i in range(28):
    	for j in range(28):
    		if(round(img[i][j]/255,2)<=0.02):
    			img[i][j]=0

    #reshaping to support our model input and normalizing
    img = np.reshape(img, (784, 1))
    activation = img
    activations = [img] 
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        activation = sigmoid(z)
        activations.append(activation)

    act=[float(activations[-1][i]) for i in range(10)]
    print(act)
    return np.argmax(activations[-1])


def sigmoid(z):
	z = np.array(z, dtype=np.float64)
	return 1.0/(1.0+np.exp(-z))


class App(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)
		self.x = self.y = 0

		# Creating elements
		self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
		self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
		self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting) 
		self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
		
		# Grid structure
		self.canvas.grid(row=0, column=0, pady=2, sticky=W,)
		self.label.grid(row=0, column=1,pady=2, padx=2)
		self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
		self.button_clear.grid(row=1, column=0, pady=2)
		# self.canvas.bind("<Motion>", self.start_pos)
		self.canvas.bind("<B1-Motion>", self.draw_lines)


	def clear_all(self):
		self.canvas.delete("all")

	def classify_handwriting(self):
		HWND = self.canvas.winfo_id() # get the handle of the canvas
		rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
		im = ImageGrab.grab(rect)
		digit = predict_digit(im)
		self.label.configure(text= "Digit : " + str(digit)) # +', '+ str(int(acc*100))+'%'
	
	def draw_lines(self, event):
		self.x = event.x
		self.y = event.y
		r=8
		self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
