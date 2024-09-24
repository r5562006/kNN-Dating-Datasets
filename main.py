# -*- coding: UTF-8 -*-

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

"""
kNN分類器

參數:
    inX - 待分類數據（測試集）
    dataSet - 訓練數據集
    labels - 分類標籤
    k - kNN算法參數，選擇最近的k個點
返回:
    sortedClassCount[0][0] - 分類結果
"""
def classify0(inX, dataSet, labels, k):
    # 獲取數據集大小
    dataSetSize = dataSet.shape[0]
    # 計算差異矩陣
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 計算平方差
    sqDiffMat = diffMat**2
    # 計算歐幾里得距離
    distances = np.sqrt(sqDiffMat.sum(axis=1))
    # 獲取排序索引
    sortedDistIndices = distances.argsort()
    # 計算類別計數
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 根據計數進行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

"""
打開並解析文件，進行分類

參數:
    filename - 文件名
返回:
    returnMat - 特徵矩陣
    classLabelVector - 分類標籤向量
"""
def file2matrix(filename):
    # 打開文件
    with open(filename, 'r', encoding='utf-8') as fr:
        arrayOLines = fr.readlines()
        arrayOLines[0] = arrayOLines[0].lstrip('\ufeff')
        numberOfLines = len(arrayOLines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0

        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            # 根據標籤進行分類
            if listFromLine[-1] == 'didntLike':
                classLabelVector.append(1)
            elif listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
            elif listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
            index += 1
    return returnMat, classLabelVector

"""
數據可視化

參數:
    datingDataMat - 特徵矩陣
    datingLabels - 分類標籤
返回:
    無
"""
def showdatas(datingDataMat, datingLabels):
    # 設定字體
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = ['black' if i == 1 else 'orange' if i == 2 else 'red' for i in datingLabels]

    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=0.5)
    axs[0][0].set_title(u'每年獲得的飛行常客里程數與玩視頻遊戲所消耗時間占比', FontProperties=font)
    axs[0][0].set_xlabel(u'每年獲得的飛行常客里程數', FontProperties=font)
    axs[0][0].set_ylabel(u'玩視頻遊戲所消耗時間占比', FontProperties=font)

    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    axs[0][1].set_title(u'每年獲得的飛行常客里程數與每周消耗的冰淇淋公升數', FontProperties=font)
    axs[0][1].set_xlabel(u'每年獲得的飛行常客里程數', FontProperties=font)
    axs[0][1].set_ylabel(u'每周消耗的冰淇淋公升數', FontProperties=font)

    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    axs[1][0].set_title(u'玩視頻遊戲所消耗時間占比與每周消耗的冰淇淋公升數', FontProperties=font)
    axs[1][0].set_xlabel(u'玩視頻遊戲所消耗時間占比', FontProperties=font)
    axs[1][0].set_ylabel(u'每周消耗的冰淇淋公升數', FontProperties=font)

    # 設定圖例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')

    for ax in axs.flatten():
        ax.legend(handles=[didntLike, smallDoses, largeDoses])

    plt.show()

"""
數據歸一化

參數:
    dataSet - 特徵矩陣
返回:
    normDataSet - 歸一化後的特徵矩陣
    ranges - 數據範圍
    minVals - 數據最小值
"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minVals, (m, 1))) / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

"""
分類器測試函數

參數:
    無
返回:
    無
"""
def datingClassTest():
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    hoRatio = 0.10
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("分類結果:%s\t真實類別:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("錯誤率:%f%%" % (errorCount / float(numTestVecs) * 100))

"""
通過三維特徵進行分類輸出

參數:
    無
返回:
    無
"""
def classifyPerson():
    resultList = ['討厭', '有些喜歡', '非常喜歡']
    precentTats = float(input("玩視頻遊戲所耗時間百分比:"))
    ffMiles = float(input("每年獲得的飛行常客里程數:"))
    iceCream = float(input("每周消耗的冰淇淋公升數:"))
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, precentTats, iceCream])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print("你可能%s這個人" % (resultList[classifierResult - 1]))

"""
主函數

參數:
    無
返回:
    無
"""
if __name__ == '__main__':
    datingClassTest()
