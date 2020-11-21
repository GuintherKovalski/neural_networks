import numpy as np
import matplotlib.pyplot as plt
from math import factorial as f
  

x = np.linspace(-2*np.pi,2*np.pi,100)
y = np.cos(x)

def taylor(x,n):
    y=0
    for i in range(n):
        y+=x**i/f(i)
    return y 

def taylor(x,n):
    y=0
    for i in range(n):
        y+=(-1)**i*x**(2*i)/f(2*i)
    return y 

plt.figure(figsize=(9, 4))
plt.scatter(x,y,color='blue')
for i in range(1,30):
    y_hat = taylor(x,i)    
    plt.plot(x,y_hat,color='red', alpha=0.09*i)
    plt.ylim(-1.1,1.1)


i = 80
x = np.linspace(-2*np.pi,2*np.pi,i)
y = (np.cos(x[:-1])-np.cos(x[1:]))/(4*np.pi/i)
plt.scatter(x[1:],y)
plt.plot(x,np.sin(x))

 


#plt.scatter(x[:len(diff(x,y))],diff(x,y))
###############################################################################

points = 1000
a=0
x = np.linspace(-3*np.pi,3*np.pi,points)
y = np.cos(x)#*x-np.sin(-x)#np.cos(x)
def diff(x,y,n,points):
    if n == 0:
        dy = y
    if n>0:
        for i in range(n):
            dx = ((abs(x[0]-x[-1]))/points-n)
            y = (y[1:]-y[:-1])/dx
    return y

##############################################################################
x = np.linspace(-3*np.pi,3*np.pi,points)
y = np.cos(x)
dy = -np.sin(x)
def dxdy(x,y,order): 
    for k  in range(order):
        dx = (x[-1]-x[0])/len(x)
        if k == 0:
            dydx = y
        elif k % 2 == 0:
            dydx = (dydx[1:]-dydx[:-1])/dx
            x = x[:-1]       
        elif k % 2 != 0:
            dydx = (dydx[1:]-dydx[:-1])/dx
            x = x[1:]
    return dydx
            
#y_hat = dxdy(x,y,order=2)
#plt.scatter(x[:len(y_hat)],y_hat)
#plt.plot(x,y)
#plt.plot(x,dy)
#x = np.linspace(-3*np.pi,3*np.pi,points)
#y = np.cos(x)
#yhat = diff(x,y,2,points)
#plt.plot(x,y)
#plt.plot(x[:len(yhat)],yhat)

########################## BOAAAAAA ##########################################

def dxdy(x,y,order): 
    dy = y
    for k  in range(order+1):
        print(k)
        dx = (x[-1]-x[0])/len(x)
        if k == 1:
            dy = y
        elif k % 2 == 0:
            dy = (dy[1:]-dy[:-1])/dx
            x = x[:-1]       
        elif k % 2 != 0:
            dy = (dy[1:]-dy[:-1])/dx
            x = x[1:]
    return dy

def taylor(x,y,n):
    a = x[int(len(x)/2)+1]
    center = int(len(x)/2)+1
    #plt.plot(y)
    #plt.ylim(min(y),max(y))
    for k in range(n+1):
        print(k)
        if k == 0:
            y_hat = (y[center]*((x-a)**k))/f(k)
            #plt.plot(y_hat)
        else:
            y_hat += (dxdy(x,y,k+1)[center]*((x-a)**k))/f(k)
            #plt.plot(y_hat)
        #plt.plot(y)
    return y_hat

points = 100
x = np.linspace(-3*np.pi,3*np.pi,points)
y = 1/(1+np.exp(-x))
y = np.sin(x)#*x#(x**4)
center = int(points/2)
for k in range(21):
    y_hat = taylor(x,y,k)
    plt.figure(figsize=(8,4))
    plt.ylim(min(y)*1.1,max(y)*1.1)
    plt.xlim(min(x),max(x))
    plt.plot(x,y)
    plt.plot(x,y_hat,c='red')
    plt.legend(['sin(x)','taylor, k= '+str(k)],loc='upper right')
    plt.title('sin(x)') 
    plt.savefig('sin'+str(k)+'.png')
       
 
###############################################################################

points = 10000
a=0
x = np.linspace(-3*np.pi,3*np.pi,points)
y = (x)**3      
plt.plot(x,y)     
plt.scatter(x,diff(x,y,0)) 
plt.scatter(x[2:-1],diff(x,y,3))    
plt.plot(x,3*x**2)







import math

x = 2
e_to_2 = x**0/math.factorial(0) + x**1/math.factorial(1) + x**2/math.factorial(2) + x**3/math.factorial(3) + x**4/math.factorial(4)
print(e_to_2)
print(math.exp(2))

import math

x = 2
e_to_2 = 0
for i in range(5):
    e_to_2 += x**i/math.factorial(i)
    
print(e_to_2)

import math

x = 2
e_to_2 = 0
for i in range(10):
    e_to_2 += x**i/math.factorial(i)
    
print(e_to_2)


import math

def func_e(n):
    x = 2
    e_to_2 = 0
    for i in range(n):
        e_to_2 += x**i/math.factorial(i)
    
    return e_to_2

out = func_e_to_2(10)
print(out)



import math

out = func_e(2,10)
print(out)

def func_e(x, n):
    e_approx = 0
    for i in range(n):
        e_approx += x**i/math.factorial(i)
    
    return e_approx

import math

x = 5
for i in range(1,20):
    e_approx = func_e(x,i)
    e_exp = math.exp(x)
    e_error = abs(e_approx - e_exp)
    if e_error < 1:
        break

import math

def func_cos(x, n):
    cos_approx = 0
    for i in range(n):
        coef = (-1)**i
        num = x**(2*i)
        denom = math.factorial(2*i)
        cos_approx += ( coef ) * ( (num)/(denom) )
    
    return cos_approx

angle_rad = (math.radians(45))
out = func_cos(angle_rad,20)
print(out)

out = math.cos(angle_rad)
print(out)


import math
import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
%matplotlib inline

angles = np.arange(-2*np.pi,2*np.pi,0.1)
p_cos = np.cos(angles)
t_cos = [func_cos(angle,3) for angle in angles]

fig, ax = plt.subplots()
ax.plot(angles,p_cos)
ax.plot(angles,t_cos)
ax.set_ylim([-5,5])
ax.legend(['cos() function','Taylor Series - 3 terms'])

plt.show()










