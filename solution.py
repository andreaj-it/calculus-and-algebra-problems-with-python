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

#### ## Linear Algebra
#Exercise 1 : Sum of two matrices
# importing numpy as np
import numpy as np
 
# creating first matrix
a=np.array([[1,2],[3,4]])
 
# creating second matrix
b=np.array([[4,5],[6,7]])
 
#print elements
print(a)
print(b)
 
# adding two matrix
#suma=a+b
#print(suma)
print(np.add(a, b))

### Exercise 2: Sum of two lists
# Use list comprehensions to perform addition of the two lists:

# initializing lists
lista1 = [2, 5, 4, 7, 3]
lista2 = [1, 4, 6, 9, 10]

 
# printing original lists
print('Lista Original 1 '+str(lista1))
print('Lista Original 2 '+str(lista2))

# using list comprehension to add two list 
suma = [ lista1[x] + lista2[x] for x in range (len (lista1))]  
 
# printing resultant list 
print('Suma de lista1 mas lista 2: '+str(suma))

# Use map() + add():
from operator import add

# initializing lists
lista1 = [2, 5, 4, 7, 3]
lista2 = [1, 4, 6, 9, 10]
 
# printing original lists
print('Lista Original 1 '+str(lista1))
print('Lista Original 2 '+str(lista2))
 
# using map() + add() to add two list 
# use map() function with add operator to add the elements of the lists lt1 and lt2  
suma = list( map (add, lista1, lista2)) # pass the lt1, lt2 and add as the parameters  
 
# printing resultant list 
print('Suma de lista1 mas lista 2: '+str(suma))

# Use zip() + sum():
from operator import add

# initializing lists
lista1 = [2, 5, 4, 7, 3]
lista2 = [1, 4, 6, 9, 10]
 
# printing original lists
print('Lista Original 1 '+str(lista1))
print('Lista Original 2 '+str(lista2))

 
# Using zip() + sum() to add two list 
# use the zip() function and sum() function to group the lists add the lists' lt1 and lt2 with index #wise.
suma = [sum(i) for i in zip(lista1, lista2 )]  
 
# printing resultant list 
print('Suma de lista1 mas lista 2: '+str(suma))

### Exercise 3 : Dot multiplication

# Import libraries
import numpy as np
 
# input two matrices
matriz1 = [[1,7,3],
 [ 4,5,2],
 [ 3,6,1]]
matriz2 = [[5,4,1],
 [ 1,2,3],
 [ 4,5,2]]
 
# This will return dot product
product=np.dot(matriz1,matriz2)

# print resulted matrix
print ('Multiplicacion ' + str(product))


