import cv2
import numpy as np

#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i,x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks
 
 
 
def remove_plate_upanddown_border(card_img):
    """
    这个函数将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
    输入： card_img是从原始图片中分割出的车牌照片
    输出: 在高度上缩小后的字符二值图片
    """
    # plate_Arr = cv2.imread(card_img)
    plate_Arr = card_img
    plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv2.threshold( plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    row_histogram = np.sum(plate_binary_img, axis=1)   #数组的每一行求和
    row_min = np.min( row_histogram )
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    #接下来挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1]-wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    #cv2.imshow("plate_binary_img", plate_binary_img)
 
    return  plate_binary_img

def distEclud (vecA, vecB):
    """
    计算两个坐标向量之间的街区距离 
    """
    return np.sum(abs(vecA - vecB))
 
def randCent( dataSet, k):
    n = dataSet.shape[1]  #列数
    centroids = np.zeros((k,n)) #用来保存k个类的质心
    for j in range(n):
        minJ = np.min(dataSet[:,j],axis = 0)
        rangeJ = float(np.max(dataSet[:,j])) - minJ
        for i in range(k):
            centroids[i:,j] = minJ + rangeJ * (i+1)/k
    return centroids
 
def kMeans (dataSet,k,distMeas = distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m,2))  #这个簇分配结果矩阵包含两列，一列记录簇索引值，第二列存储误差。这里的误差是指当前点到簇质心的街区距离
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[ np.nonzero(clusterAssment[:,0]==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)
    return centroids , clusterAssment
 
 
 
def biKmeans(dataSet,k,distMeas= distEclud):
    """
    这个函数首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
    输入：dataSet是一个ndarray形式的输入数据集 
          k是用户指定的聚类后的簇的数目
         distMeas是距离计算函数
    输出:  centList是一个包含类质心的列表，其中有k个元素，每个元素是一个元组形式的质心坐标
            clusterAssment是一个数组，第一列对应输入数据集中的每一行样本属于哪个簇，第二列是该样本点与所属簇质心的距离
    """
    m = dataSet.shape[0]
    clusterAssment =np.zeros((m,2))
    centroid0 = np.mean(dataSet,axis=0).tolist()
    centList = []
    centList.append(centroid0)
    for j in range(m):
         clusterAssment[j,1] = distMeas(np.array(centroid0),dataSet[j,:])**2
    while len(centList) <k:       #小于K个簇时
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0] == i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = np.sum(splitClustAss[:,1])
            sseNotSplit = np.sum( clusterAssment[np.nonzero(clusterAssment[:,0]!=i),1])
            if (sseSplit + sseNotSplit) < lowestSSE:         #如果满足，则保存本次划分
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0] ==1)[0],0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0,:].tolist()
        centList.append( bestNewCents[1,:].tolist())
        clusterAssment[np.nonzero(clusterAssment[:,0] == bestCentTosplit)[0],:] = bestClustAss
    return centList, clusterAssment
 
 
def split_licensePlate_character(plate_binary_img):
    """
    此函数用来对车牌的二值图进行水平方向的切分，将字符分割出来
    输入： plate_gray_Arr是车牌的二值图，rows * cols的数组形式
    输出： character_list是由分割后的车牌单个字符图像二值图矩阵组成的列表
    """
    plate_binary_Arr = np.array ( plate_binary_img )
    row_list,col_list = np.nonzero (  plate_binary_Arr >= 255 )
    dataArr = np.column_stack(( col_list,row_list))   #dataArr的第一列是列索引，第二列是行索引，要注意
    centroids, clusterAssment = biKmeans(dataArr, 7, distMeas=distEclud)
    centroids_sorted = sorted(centroids, key=lambda centroid: centroid[0])
    split_list =[]
    for centroids_ in  centroids_sorted:
        i = centroids.index(centroids_)
        current_class = dataArr[np.nonzero(clusterAssment[:,0]==i)[0],:]
        x_min,y_min = np.min(current_class,axis =0 )
        x_max, y_max = np.max(current_class, axis=0)
        split_list.append([y_min, y_max,x_min,x_max])
    character_list = []
    for i in range(len(split_list)):
        single_character_Arr = plate_binary_img[split_list[i][0]: split_list[i][1], split_list[i][2]:split_list[i][3]]
        character_list.append( single_character_Arr )
        cv2.imshow('character'+str(i),single_character_Arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    return character_list              #character_list中保存着每个字符的二值图数据

image = cv2.imread("./imgs/out2.png")
bin = remove_plate_upanddown_border(image)
# split_licensePlate_character(bin)
cv2.imshow("bin", bin)
cv2.waitKey()

# Gaussian = cv2.GaussianBlur(image, (3, 3), 0)
# cv2.imshow("Gaussian", Gaussian)
# cv2.waitKey()

# gray = cv2.cvtColor(Gaussian, cv2.COLOR_RGB2GRAY)
# cv2.imshow("gray", gray)
# cv2.waitKey()

# ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# cv2.imshow("bin", bin)
# cv2.waitKey()


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# erode = cv2.erode(bin, kernel)
# cv2.imshow("erode", erode)
# cv2.waitKey()

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# dilate = cv2.dilate(bin, kernel)
# cv2.imshow("dilate", dilate)
# cv2.waitKey()


# contours, hierarchy = cv2.findContours(
#     dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )
# # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
# # cv2.imshow("image", image)
# # cv2.waitKey()

# chars = []
# for item in contours:
#     rect = cv2.boundingRect(item)
#     x, y, w, h = cv2.boundingRect(item)
#     chars.append(rect)
#     # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)

# chars = sorted(chars, key=lambda s: s[0], reverse=False)
# plateChars = []
# for i, word in enumerate(chars):
#     x, y, w, h = word
#     plateChar = image[y : y + h, x : x + w]
#     cv2.imshow("image" + str(i), plateChar)
#     cv2.waitKey()
#     if (h > (w * 1.5)) and (h < (w * 8)) and (w > 3):
#         plateChar = image[y : y + h, x : x + w]
#         plateChars.append(plateChar)

# # for i, im in enumerate(plateChars):
# #     cv2.imshow("char" + str(i), im)
# #     cv2.waitKey()

# cv2.destroyAllWindows()
