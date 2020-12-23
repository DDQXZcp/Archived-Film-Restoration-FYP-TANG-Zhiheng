from tensorflow import keras
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def generate_map(frame,num,threshold,model_name="mymodel4_0.985", save_flag=False):
    num_frame = "frame"+ str(frame)+ "_"+ str(num) + ".jpg"
    width = 600
    height = 600
    channel = 1
    # path='D:\pycharm project'
    save_path = 'D:\dataset\CNN_out\\'
    #ori_files = os.listdir('D:\FYP\original')  # D:\FYP\original  D:\FYP\\test_o
    ori_files = os.listdir('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\test_o')

    '''original file, read as size of (600,600,1)'''
    a = np.zeros((1, width, height, channel))
    e = np.zeros((width, height, channel))
    c = np.array(io.imread('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\test_o\\' + num_frame)) #ori_files[num_frame]
    #print('D:\FYP\original\\' + ori_files[num_frame])
    c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    save_name = "./output/" + "frame" + str(frame) + "_" + str(num) + "-1-Original.jpg"
    io.imsave(save_name, c)
    e[:, :, 0] = c
    a[0] = e / 255
    x_test = a
    #plt.imshow(c)
    #plt.show()
    '''#truth file
    gt_path = os.listdir('D:\FYP\\truth')
    b = np.zeros((1, 600, 600))
    d = np.array(io.imread('D:\FYP\\truth\\'+gt_path[num_frame], as_gray=True))
    b[0]=d
    y_test=b
    plt.imshow(d)
    plt.show()
    '''

    model = keras.models.load_model(model_name)
    predict_scores = model.predict(x_test)
    print("Output max:", predict_scores.max())
    print("Output min:", predict_scores.min())

    '''map the 0-1 pixel value to 0-255'''
    m = predict_scores[0] * 255
    save_name = "./output/" + "frame" + str(frame) + "_" + str(num) + "-2-Score.jpg"
    demo = cv2.equalizeHist(255-m.astype(np.uint8))
    io.imsave(save_name, demo)
    #plt.imshow(m)
    #plt.show()
    hist1 = np.histogram(m[:, :, 0], bins=256)
    print(hist1)

    '''set threshold to select the possible defect pixels'''
    n = np.zeros((width, height),dtype='uint8')
    for i in range(width):
        for j in range(height):
            if m[i][j][0] < threshold:
                n[i][j] = 255
            else:
                n[i][j] = 0
    #plt.imshow(n)
    #plt.show()

    #save_name = save_path + str(num_frame) + ".jpg"
    save_name = "./output/"+"frame" + str(frame) + "_" + str(num) + "-3-InitialDetection.jpg"
    io.imsave(save_name, n)

    truth = np.array(io.imread('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\test_t\\' + num_frame))  # ori_files[num_frame]
    save_name = "./output/" + "frame" + str(frame) + "_" + str(num) + "-GroundTruthLabel.jpg"
    io.imsave(save_name, truth)

    return c, n, truth

#1045 1
#1002_4
#1073_1
frame = 1012
num = 2
threshold=1.5
original_current, map1_address, truth_current = generate_map(frame, num, threshold, save_flag=True)

#previous_frame=num_frame-6
#threshold=3
threshold=1.5
original_next, map2_address, truth_next =generate_map(frame+1, num, threshold)
'''
map1=io.imread(map1_address)
map2=io.imread(map2_address)

for i in range(map1.shape[0]):
    for j in range(map1.shape[1]):
        if map1[i][j]==map2[i][j]:
            map1[i][j]=0

io.imshow(map1)
io.show()
'''

for i in range(map1_address.shape[0]):
    for j in range(map1_address.shape[1]):
        if map2_address[i][j]==map1_address[i][j]:
            map1_address[i][j]=0

#io.imshow(map1_address)
#io.show()

save_name = "./output/"+"frame" + str(frame) + "_" + str(num) + "-4-AfterEliminate.jpg"
io.imsave(save_name, map1_address)

kernel = np.ones((7,7),np.uint8)
dilated_map = cv2.dilate(map1_address.astype(np.uint8),kernel,iterations = 1)
dst = cv2.inpaint(original_current.astype(np.uint8), dilated_map.astype(np.uint8),3,cv2.INPAINT_TELEA)

save_name = "./output/"+"frame" + str(frame) + "_" + str(num) + "-5-Restored.jpg"
io.imsave(save_name, dst)

#plt.imshow(n)
#plt.show()

original = cv2.imread('./output/frame' + str(frame) + '_' + str(num) + '-1-Original.jpg', 1)
detect = cv2.imread('./output/frame' + str(frame) + '_' + str(num) + '-4-AfterEliminate.jpg', 0)
truth = cv2.imread('./output/frame' + str(frame) + '_' + str(num) + '-GroundTruthLabel.jpg', 0)

print(original.shape)
print(detect.shape)
print(truth.shape)
real_detect = 0
missing = 0
false_alarm = 0
for i in range(0, 600):
    for j in range(0, 600):
        if (truth[i][j] > 0):
            if (detect[i][j] > 0):
                original[i][j][0] = 0
                original[i][j][1] = 255
                original[i][j][2] = 0
                real_detect = real_detect + 1
            else:
                original[i][j][0] = 0
                original[i][j][1] = 255
                original[i][j][2] = 255
                missing = missing + 1
        elif (detect[i][j] > 0):
            false_alarm = false_alarm + 1
            original[i][j][0] = 0
            original[i][j][1] = 0
            original[i][j][2] = 255

print(real_detect)
print(missing)

cv2.imwrite('./output/frame' + str(frame) + "_" + str(num) + '-6-accuracy.jpg', original)

