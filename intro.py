import numpy as np
import random
import matplotlib.pyplot as plt
import time

# print("Print this")
# test = 2
# print("Print not this")
#
# 2+2
# 8-5
# 5/2
# 5//2
# 8**0.5
# np.sqrt(9)
# np.pi
# np.e
#
# round(2.6)
# 1j + 2
# np.sqrt(-1.0+0j)

# Numpy array:
array1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
array1[1][1]
array1[1,1]
np.append(array1,np.array([[1],[23],[0]]),axis=1)

### Vectors

vect1 = np.array([1,2,3])
vect2 = np.array([2,4,np.sqrt(5)])

np.dot(vect1,vect2)

np.cross(vect1,vect2)



#length of a vector
# Three different ways to calculate the norm of the vector

# vect2 = np.array([1,2,3,4,5,6,7,8,9])

set1 = np.array([])
set2 = np.array([])
set3 = np.array([])
x = np.arange(1001)

for n in x:
    vect2 = np.random.randint(100,size=n)
    
    #1
    start = time.time()
    np.linalg.norm(vect2)
    set1 = np.append(set1,100000*(time.time()-start))
    
    #2
    start = time.time()
    np.sqrt(np.sum(vect2**2))
    set2 = np.append(set2,100000*(time.time()-start))
    
    start = time.time()
    sum = 0
    for s in range(vect2.size):
        sum += vect2[s]**2
    np.sqrt(sum)
    set3 = np.append(set3,100000*(time.time()-start))

plt.plot(x,set1,label='np.linalg.norm')
plt.plot(x,set2,label='np.sqrt(np.sum(vect**2))')
plt.plot(x,set3,label='traditional')
plt.legend()
#The result shows that numpy package takes more time, this an anomaly
#Resolved: for bigger array, the calculation time is way smaller
# Let's plot the difference in performance

### Matrices

mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[2,2],[2,2]])
test = mat1*mat2
test =np.matmul(mat1,mat2)


### Dictonary

dict1 = {
        "Particule" : "Neutron",
        "Mass" : 10,
        "Charge" : 1,
        }

dict1
dict1.keys()
dict1.values()

### Save and Load

storelist = [1,2,3,4,5]

np.savetxt("test.txt", storelist)

file = open('storelist3.txt','w')
for i in storelist:
    file.write(str(i**2))
    file.write('\n')
file.close()

with open('storelist3.txt','w') as file:
    for i in storelist:
        file.write(str(i**2))
        file.write('\n')
loadlist = np.loadtxt("storelist3.txt")

loadlist

plt.scatter([1,2,3,4,5],[1,2,3,4,5])


coords1 = np.array([[1,2,3],[3,4,5],[6,2,-5],[4,5,2]])
x1, y1, z1 = coords1.T #Transpose
plt.xlabel('x')
plt.ylabel('y,z')
plt.scatter(x1,y1)
plt.scatter(x1,z1)



x1 = np.linspace(0,4,110)
y1 = 1 + 2*np.cos(3*x1)

plt.xlim(0,8)
# plt.ylim(-3,5)
plt.plot(x1,y1)


x2 = np.linspace(4,8,110)
y2 = 1 + 2*np.cos(3*x1)
plt.plot(x2,y2)

plt.savefig('testplot.png')


# Density Plot
np.meshgrid(
np.linspace(0,5,6),
np.linspace(5,10,6)
)

np.transpose(
np.meshgrid(
np.linspace(0,5,6),
np.linspace(5,10,6)
)
)
x2, y2 = np.meshgrid(
np.linspace(-10,10,201),
np.linspace(-10,10,201)
)

z2= x2 + y2**2
z2

contourplot = plt.contour(x2,y2,z2)
plt.clabel(contourplot, inline = 1, fontsize = 10)

x3, y3 = np.meshgrid(
np.linspace(-10,10,201),
np.linspace(-10,10,201)
)

z3 = np.cos(x3) + np.sin(y3)
# z3 = np.cos(x3+y3)+ 0.05*(x3-y3)

plt.contourf(x3,y3,z3)
plt.colorbar()

proj = plt.axes(projection='3d')
proj.view_init(20,70)
proj.contour3D(x3,y3,z3,100)#,cmap='binary')


proj2 = plt.axes(projection='3d')
z1 = np.linspace(0,30,301)
x1 = np.sin(z1)
y1 = 2*np.cos(z1)
plt.xlim(-2,2)
plt.ylim(-2,2)
proj2.plot3D(x1,y1,z1)

































