from random import randint
import time

class Vehicle:
    tracks=[]
    def __init__(self,i,xi,yi,area,max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.area = area
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        self.state ='0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
        self.type = 'normal'

    def getRGB(self):  #For the RGB colour
        return (self.R,self.G,self.B)
    def getTracks(self):
        return self.tracks

    def getId(self): #For the ID
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):  #for x coordinate
        return self.x

    def getY(self):  #for y coordinate
        return self.y

    def getArea(self):
        return self.area

    def updateArea(self, area):
        self.area = area

    def getType(self):
        return self.type

    def setType(self, type):
        self.type = type

    def updateType(self, treshold):
        if self.area > treshold:
            self.type = 'heavy'
        else:
            self.type = 'normal'

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_UP(self, detection_line):
        if len(self.tracks)>=2:
            if self.state=='0':
                if self.tracks[-1][1]<detection_line and self.tracks[-2][1]>=detection_line:
                    self.state='1'
                    self.dir='up'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def going_DOWN(self, detection_line):
        if len(self.tracks)>=2:
            if self.state=='0':
                if self.tracks[-1][1]>detection_line and self.tracks[-2][1]<=detection_line:
                    self.state='1'
                    self.dir='down'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age+=1
        if self.age>self.max_age:
            self.done=True
        return  True

    def getAge(self):
        return self.age

