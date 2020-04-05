# import all required libraries

import math
import keras
import random
import tensorflow as tf
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import pickle
import numpy as np
from Network import *
from misc import sigmoid


# model = pickle.load(open('mnist', 'rb'))
# print(model.biases)
biases = pickle.load(open('biases.txt', 'rb'))
# print(biases)
biases[0] = tf.reshape(biases[0], (30,28))
biases[1] = tf.reshape(biases[1], (10,28))
# print(biases)
weights = pickle.load(open('weights.txt', 'rb'))
weights[0] = tf.reshape(weights[0], (30,28))
weights[1] = tf.reshape(weights[1], (30,10))
# for i in weights:
# 	i = np.array(i)
# print(weights)

# net = Network([28,30,10])

def predict_digit(img):
	#resize image to 28x28 pixels
    img = img.resize((28,28))

    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype = tf.float32)

    #reshaping to support our model input and normalizing
    img = tf.reshape(img, (1, 28, 28))
    img = img/255.0
    # print(img)

    #predicting the class
    # res = predict(img)
    # print(type(res[0]))
    # print(res)
    # result = []
    # for i in res:
    # 	print(next(i))
    # print(result)
    # print(np.argmax(res))
    test_results = []

    for x in img:
    	for b,w in zip(biases, weights):
    		print(w.shape)
    		print(x.shape)
    		test_results = [sigmoid(tf.tensordot(w,x,1)+b)]

    # test_results = [feedForward_test(x, weights, biases) for x in img]
    print(test_results)

    # t = tf.argmax(test_results)
    # tf.compat.v1.disable_eager_execution()
    # print(tf.executing_eagerly())
    # ans = tf.compat.v1.get_default_session().run(t)
    # ans = t.numpy()
    # with tf.compat.v1.Session() as sess:
    # 	ans = sess.run(t)

    # print(ans)
    return 0

def predict(img):
	print("hello")
	test_results = [(net.feedForward_test(x, weights, biases) for x in img)]
	# print(type(test_results[0]))
	yield test_results

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
