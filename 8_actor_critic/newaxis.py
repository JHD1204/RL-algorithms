import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.shape)
print(a)
aa=a[:,np.newaxis]
print(aa.shape)
print (aa)

aaa=a[np.newaxis]
print(aaa.shape)
print (aaa)