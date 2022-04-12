#!/usr/bin/env python
# coding: utf-8

# In[1]:


#######벡터, 행렬 연산, 그래프 그리기 201811194 이진욱######

import numpy as np
import sys
print "Python version : ", sys.version


# In[2]:


#프린트
def print_val(x):
    print "Type : ", type(x)
    print "Shape : ", x.shape
    print "값 : \n", x
    print " "


# In[3]:


print("rank 1 np array\n")
x = np.array([1, 2, 3])
print_val(x)

x[0] = 5
print_val(x)


# In[4]:


print("rank 2 np array\n")
y = np.array([[1, 2, 3], [4, 5, 6]])
print_val(y)

print("rank 2 ones\n")
a = np.ones((3,2))
print_val(a)

print("rank 2 zeros\n")
a = np.zeros((2,2))
print_val(a)

print("rank 2 identity matrix\n")
a = np.eye(3,3)
print_val(a)


# In[5]:


print("랜덤 행렬 - uniform\n")
a = np.random.random((4,4))
print_val(a)

print("랜덤 행렬 - Gaussian\n")
a = np.random.randn(4,4)
print_val(a)


# In[6]:


print("np array indexing\n")
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_val(a)

b = a[:2, 1:3]
print_val(b)

print("행렬의 n번째 행 얻기\n")
row1 = a[1, :]
print_val(row1)


# In[7]:


print("행렬의 원소별 연산\n")
m1 = np.array([[1,2], [3,4]], dtype=np.float64)
m2 = np.array([[5,6], [7,8]], dtype=np.float64)

print("ADD\n")
print_val(m1 + m2)
print_val(np.add(m1, m2))


print("SUBTRACT\n")
print_val(m1 - m2)
print_val(np.subtract(m1, m2))


print("PRODUCT\n")
print_val(m1 * m2)
print_val(np.multiply(m1, m2))


print("DIVISION\n")
print_val(m1 / m2)
print_val(np.divide(m1, m2))


print("SQURE ROOT\n")
print_val(np.sqrt(m1))


# In[8]:


print("행렬 연산\n")
m1 = np.array([[1,2], [3,4]])
m2 = np.array([[5,6], [7,8]])
v1 = np.array([9,10])
v2 = np.array([11,12])

print_val(m1)
print_val(m2)
print_val(v1)
print_val(v2)


# In[9]:


print("벡터-벡터 연산\n")
print_val(v1.dot(v2))
print_val(np.dot(v1,v2))


# In[10]:


print("벡터-행렬 연산\n")
print_val(m1.dot(v1))
print_val(np.dot(m1,v1))


# In[11]:


print("행렬-행렬 연산\n")
print_val(m1.dot(m2))
print_val(np.dot(m1,m2))


# In[12]:


print("전치 행렬\n")
print_val(m1)
print_val(m1.T)


# In[13]:


print("합\n")
print_val(np.sum(m1))

print("압축\n")
print_val(np.sum(m1, axis=0))
print_val(np.sum(m1, axis=1))

m1 = np.array([[1,2,3],[4,5,6]])
print_val(m1)

print_val(np.sum(m1))
print_val(np.sum(m1, axis=0))
print_val(np.sum(m1, axis=1))


# In[14]:


print("zeros-like\n")
m1 = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
m2 = np.zeros_like(m1)
print_val(m1)
print_val(m2)


# In[15]:


print("Matplot library\n")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.arange(0,10,0.1)
y = np.sin(x)

plt.plot(x,y)


# In[16]:


print("한 번에 두 개 그래프 그리기\n")

y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('sin and cos')
plt.legend(['sin', 'cos'])

plt.show()


# In[17]:


print("Subplot\n")
plt.subplot(2,1,1)
plt.plot(x, y_sin)
plt.title('sin')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('cos')

plt.show()


# In[ ]:




