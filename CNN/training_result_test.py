from tensorflow import keras
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
'''
mymodel4_30: 30:0.5
mymodel4_50epoch:50epoch 0.98:0.2
'''


path='D:\pycharm project'
#ori_files= os.listdir('D:\FYP\original') #D:\FYP\original  D:\FYP\\test_o
ori_files= os.listdir('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\test_o') #D:\FYP\original  D:\FYP\\test_o
#gt_path = os.listdir('D:\FYP\\truth')
gt_path = os.listdir('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\test_t')
a = np.zeros((1, 600, 600, 1))
b = np.zeros((1, 600, 600))
#c =np.array(io.imread('D:\FYP\original\\'+ori_files[28]))
c =np.array(io.imread('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\test_o\\'+ori_files[28]))
print(ori_files[28])
c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)

e=np.zeros((600,600,1))
e[:,:,0]=c
plt.imshow(c)
plt.show()
#d = np.array(io.imread('D:\FYP\\truth\\'+gt_path[28], as_gray=True))
d = np.array(io.imread('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\test_t\\'+gt_path[28], as_gray=True))
plt.imshow(d)
plt.show()
a[0]=e/255
b[0]=d
x_test=a
y_test=b

model = keras.models.load_model("mymodel4_0.985")

#test_scores = model.evaluate(x_test, y_test)
predict_scores = model.predict(x_test)
print()
print("Output shape:",predict_scores.shape)
print("Output max:", predict_scores.max())
print("Output min:", predict_scores.min())
m=np.zeros((600,600))
m=predict_scores[0]*255

plt.imshow(m)
plt.show()

n=np.zeros((600,600))
'''
re1=map(m.index,heapq.nlargest(4,m))
print(list(re1))
'''
for i in range(600):
    for j in range(600):
        if m[i][j][0]<2:
            n[i][j]=255
        else:
            n[i][j] = 0
n1=plt.imshow(n)
n1.set_cmap('gray')
plt.show()
print("After threshold max:",m.max())

hist1=np.histogram(m[:,:,0],bins=256)
print(hist1)

#plt.hist(m[:,:,0],bins=256)
#plt.show()
