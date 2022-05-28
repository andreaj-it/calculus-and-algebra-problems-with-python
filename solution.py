#Exercice 1
#import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#define the distance function
def f(x): return 2.5*x #function

#plot the distance function on domain(t)
x = np.linspace(0, 10, 100)
plt.plot(x, f(x))

# create a dataframe
import pandas as pd
f1 = pd.DataFrame({'x': x, 'f(x)': f(x)})
f1.head()

#Exercice 2
#define and plot the quadratic funtion
def f(x): return 0.5*0.5*(x**2)
x = np.linspace(0,10)
plt.plot(x, f(x)) 
# create a dataframe
f2 = pd.DataFrame({'x': x, 'f(x)': f(x)})
f2.head()

#Exercice 3
#define and plot the funtion
def f(x): 
    return 0.1*(x)**2 -9*x + 4500

# derivad
def df(x):
    return 0.2*x - 9
    
x = np.linspace(0,100)
plt.plot(x, f(x))

def visualize(f, x=None):
    
    xArray = np.linspace(-10, 10, 100) 
    yArray = f(xArray)
    sns.lineplot(x=xArray, y=yArray)
    
    if x is not None:
        assert type(x) in [np.ndarray, list] # x should be numpy array or list
        if type(x) is list: # if it is a list, convert to numpy array
            x = np.array(x)

            
        y = f(x)
        sns.scatterplot(x=x, y=y, color='red')

def gradient_descent(x, nsteps=1):
    
    
    #collectXs is an array to store how x changed in each iteration, 
    #so we can visualize it later
    
    collectXs = [x]
    
    #learning_rate is the value that we mentioned as alpha in previous section
    
    learning_rate = 1e-01
    
    for _ in range(nsteps):
        
        #the following one line does the real magic
        #the next value of x is calculated by subtracting the gradient*learning_rate by itself
        #the intuation behind this line is in the previous section
        
        x -= df(x) * learning_rate 
        collectXs.append(x)
        
    #we return a tuple that contains
    #x -> recent x after nsteps 
    #collectXs -> all the x values that was calculated so far
    
    return x, collectXs

plt.plot(x, f(x), color='red')
visualize(f, x=[0, 100])

visualize(f, x=[0,45 ])

visualize(f, x=[0])

x= 0
x, collectedXs = gradient_descent(x, nsteps=1)
print(x)

x, collectedXs = gradient_descent(x, nsteps=1)
print(x)

x, collectedXs = gradient_descent(x, nsteps=1000)
print(x)

visualize(f, x = collectedXs)

