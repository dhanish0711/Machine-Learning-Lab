#Study and Utilize NumPy for numerical operations, array manipulation, matrix computations essential for machine learning tasks.
import numpy as np
a = np.array([(1,2,3),(3,4,4)], dtype = 'int8') # can use [] or () to Diffrentiate elements not making tuple
print(a)
print(type(a))

b=np.array([1,2,3])
print (b)

print(b.dtype) #tells us the data types of elemnt in the array

#get dimensions
print(a.ndim)
print(b.ndim)

#get shape
print(a.shape)
print(b.shape)

#get type
print(a.dtype)
print(b.dtype)

 # If we want to specify the size of it
a=np.array([(1,2,3),(3,4,4)],dtype='int16')
print(a.dtype)

#Get Itemsize, in bytes
print(a.itemsize)

#Get total size
print(a.size*a.itemsize)
print(a.nbytes)

import numpy as np
a = np.array([(1,2,3),(3,4,4)], dtype = 'int8')
print(a)
print(type(a))
b = np.array([1,2,3])
print(b)
print(type(b))
print(a.dtype)
print(b.dtype)
print(a.ndim)
print(b.ndim)
print(a.shape)
print(a.itemsize)
print(a.size)
print(a.size*a.itemsize)
print(a.nbytes)

a = np.array([(1,2,3,4,5,6,7),(8,9,10,11,12,13,14)])
print(a)
print(a.shape)
print(a[1,4])
print(a[0,1::2])

w = np.arange(start = 1, stop = 10, step = 2)
print(w)

import numpy as np
q = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(q)
print(q[0,1,1])
print(q[:,1,:1])
q[:,1,:] = [9,9],[9,9]
print(q)

a = np.zeros((2,2)) # All 0's matrix
print(a)
print(np.zeros((2,2,3)))
print(np.ones((4,2,2), dtype = 'int32'))
print(np.full((2,2),99, dtype= 'float32'))
print(np.full_like(q,7))
print(np.random.rand(2,3))
print(np.random.random_sample(q.shape))
print(np.random.randint(7,size =(3,3)))
print(np.random.randint(-6,6,size =(2,2)))
print(np.random.randint(-6,6,2))
print(np.identity(7,dtype= 'int16'))
arr = np.array([1,2,3])
z = np.repeat(arr,3,axis = 0)
print(z)
o = np.zeros([3,3])
print(o)
o[1:2] = 9
print(o)
print(o[1])

# Mathematical Operations and BroadCasting
a = np.array([1,2,3,4])
print(a)
a += 2
print(a)
print(a+2)
print(a-2)
print(a*2)
print(a/2)

a = np.ones((2,3))
print(a)
b = np.full((3,2),2)
print(b)
np.matmul(a,b)
c= np.identity(3)
np.linalg.det(c)

stats = np.array([[5,2,3],[4,5,6]])
print(stats)
print(np.max(stats, axis = 1))
print(np.max(stats, axis = 0))
print(np.min(stats))
print(np.sum(stats))
