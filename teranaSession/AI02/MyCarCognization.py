import numpy as np
import warnings #去掉不必要的警告
import  cv2 as cv
import hmmlearn.hmm as hl
import os
import pdb

def show_image(title,img):
    cv.imshow(title,img)

def read_image(imagePath):
    return cv.imread(imagePath)

#根据图像路径，获取图像对象
def get_objects(directory):
    if not os.path.isdir(directory):
        raise IOError("The direcotry '{}' is invalid".format(directory))
    objects = {}
    for current_dir, sub_dir,files in os.walk(directory):
        # print(current_dir)
        # print(sub_dir)
        # print(files)
        for jepg in (file for file in files if file.endswith('.jpg')):
            path = os.path.join(current_dir,jepg)
            label = path.split(os.path.sep)[-2]
            if label not in objects:
                objects[label] = []
            objects[label].append(path)
    print(objects)
    return objects

#定义图片数组的大小即图片大小，对图片进行同比例缩放，保证训练时 图片训练数组大小一致
#当图片长度超多size时候，对宽度进行缩放，当图片宽度超过size时候，对长度进行缩放
def resize_img(image,size):
    # print("oraginal image shape:",image.shape)
    h,w = image.shape[:2]
    scale = size / min(h,w)
    # print("h={},w={},scale={}".format(h,w,scale))
    # show_image("Orignal",image)
    image = cv.resize(image,None,fx = scale,fy = scale)
    # show_image("Scale", image)
    return image

def calc_feature(image):
    star = cv.xfeatures2d.StarDetector_create()
    keypoints = star.detect(image)
    sift = cv.xfeatures2d.SIFT_create()
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    keypoints,desc = sift.compute(gray,keypoints)
    return desc

def read_data(directory):
    objects = get_objects(directory)
    x, y ,z = [],[],[] #x, y,z  分别存放 特征值矩阵，特征值矩阵对应的lable,图片数组
    count = 0
    for lable,filenames in objects.items():
        z.append([])
        descs = np.array([])
        for file in filenames:
            count +=1
            image = read_image(file)
            # print(file)
            z[-1].append(image)
            image = resize_img(image,200)
            desc = calc_feature(image)
            descs = desc if len(descs)==0 else np.append(descs,desc,axis = 0) #数组横向增加
            # print(desc.shape)
            # print(descs.shape)
        x.append(descs)
        y.append(lable)
    return x,y,z

def read_testing_data(directory):
    objects = get_objects(directory)
    x, y, z = [], [], []  # x, y,z  分别存放 特征值矩阵，特征值矩阵对应的lable,图片数组
    for lable, filenames in objects.items():
        # z.append([])
        for file in filenames:
            image = read_image(file)
            desc = calc_feature(image)
            x.append(desc)
            z.append(image)
            # show_image(lable,z[0])
        y.append(lable)
    return x,y,z


def train_model(x,y):
    models = {}
    for lable,descs in zip(y,x):
        # n_components =4, 表示四个颜色通道分组，分别为BGR 和 阿尔法通道
        #covariance_type = diag 表示协方差采用对角线, n_iter 迭代次数
        model = hl.GaussianHMM(n_components= 4,covariance_type='diag',n_iter=3000)  #高斯隐马尔可夫模型
        model.fit(descs)
        models[lable]=model
    print(models.items())
    return models

def pred_model2(models,x):
    best_lable = []
    lables = []
    pdb.set_trace()
    for descs in x:
        scores = []
        for label, model in models.items():
            score = model.score(descs)
            scores.append(score)
            lables.append(label)
        scores = np.array(scores)
        best_score_index = scores.argmax()
        best_lable.append(lables[best_score_index])
    return best_lable


def pred_model(models,x):
    y = []
    scores = []
    results = {}
    for descs in x:
        best_lable,best_score = None,None
        for label, model in models.items():
            score = model.score(descs)
            results[score] = label
            if(best_lable is None):
                best_lable = label
            if(best_score is None):
                best_score = score
            if(best_score<score):
                best_score = score
                best_lable = label
        y.append(best_lable)
        scores.append(best_score)
        print("best lable", best_lable, scores)

    print(results)
    return y
    # for descs in x:
    #     scores = []
    #     pred_lables = []
    #     for lable, model in models.items():
    #         score = model.score(descs)
    #         scores.append(score)
    #         pred_lables.append(lable)
    #
    #     score_index = np.array(scores).argmax()
    #     y.append(pred_lables[score_index])

def show_lables(labels,pred_lables,images):
    index = 0

    # for lable, pred_lable, row in zip(labels,pred_lables, images):
    #     print(pred_lable)
    #     for image in row:
    #         index += 1
    #         # show_image("#{}:{} Predict to {} {}".format(index,lable,'==' if lable==pred_lable else '!=',pred_lable),image)
    #         show_image("#{}: This is {}, Oraginal Lable: {}".format(index, pred_lable,lable), image)

    for index, lable in enumerate(pred_lables):
        show_image("#{}: This is {}".format(index, lable), images[index])

    # for pred_lable, row in zip(pred_lables,images):
    #     for image in row:
    #         index += 1
    #         # show_image("#{}: This is {}".format(index, pred_lable), image)
    #     if(index>5):
    #         print("exceed 5, so break")
    #         break

def main():
    warnings.filterwarnings(action='ignore',category=DeprecationWarning)
    np.seterr(all='ignore')
    directory = r"E:\python\study\terana_Learning\AI\data\objects\training"
    # test_dict = r"E:\python\study\terana_Learning\AI\data\objects\testing"
    test_dict = r"E:\python\study\terana_Learning\AI\data\objects\noIdentify"
    # get_objects(directory)
    train_x,train_y,train_z =  read_data(directory)
    models = train_model(train_x, train_y)

    # test_x, test_y,test_z = read_data(test_dict)
    test_x, test_y, test_z = read_testing_data(test_dict)
    # pred_y = pred_model(models,test_x)
    pred_y = pred_model2(models,test_x)
    print("pred_y=",pred_y,"-->",test_y)


    # show_lables(test_y,pred_y,test_z)
    # show_lables(None, pred_y, test_z)
    cv.waitKey()
    cv.destroyAllWindows()
main()

