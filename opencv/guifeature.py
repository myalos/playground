import cv2 as cv

# 下面这个没用，就不纠结这个samples.findFile了
#img = cv.imread(cv.samples.findFile('starry_night.jpg'))

# 下面是显示图片的
def func1():
    img = cv.imread('1.jpg')
    print(img.shape)
    cv.imshow('display', img)
    k = cv.waitKey(0)

# 下面是视频相关的
def func2():
    # 下面这个会开启laptop的摄像头
    # 这个VideoCapture函数的参数要们是device index要么是videofile的name
    # cv.get(propId)方法 propId 是0到18的数字，每个数字代表着视频的一个属性
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('cannot open camera')
        exit()

    while True:
        ret, frame = cap.read()
        print(frame.shape)
        exit()
        if not ret:
            print("Can't receive frame. Exiting ... ")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# 这个与func2非常像
def func3(name):
    cap = cv.VideoCapture(name)
    if not cap.isOpened():
        print('cannot load file')
        exit()
    while True:
        ret, frame = cap.read()
        print(type(frame))
        print(frame.shape)
        exit()
        if not ret:
            print("error")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        cv.waitKey(1)
        #if cv.waitKey(25) == ord('q'):
        #    break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    func2()
    #func3('1.mp4')
