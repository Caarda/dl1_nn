import math

class FlyingCycle:
    def __init__(self):
        self.y = 220
        self.yspeed = 0
        self.xspeed = 70
        self.angle = 0
        self.done = False

        self.inputs = []

        self.frames = 0
        self.dist = 0

    def reset(self):
        self.y = 220
        self.yspeed = 0
        self.xspeed = 70
        self.angle = 0
        self.done = False

        self.frames = 0
        self.dist = 0

    def getState(self):
        return [self.frames, self.dist, self.y, self.xspeed, self.yspeed, self.angle, self.done]

    def nextFrame(self, input):
        if input == [1, 0, 0]:
            self.inputs.append(-1)
        elif input == [0, 1, 0]:
            self.inputs.append(0)
        elif input == [0, 0, 1]:
            self.inputs.append(1)
        self.reward = 0
        self.frames += 1

        if (self.y < 25): self.y = 25; self.yspeed = 0
        if (self.y > 430):
            self.done = True
            self.reward = self.dist/(self.frames+40)
            return self.reward, self.done, self.dist/(self.frames+40), self.inputs

        if (self.angle < -90): self.angle = -90
        if (self.angle > 90): self.angle = 90

        self.dist += round(self.xspeed)

        self.xspeed *= 0.999
        self.y += math.floor(self.yspeed / 0.05) * 0.05; self.y = round(self.y*1000)/1000
        self.yspeed += max(0, 20 - self.xspeed) / 40

        if input[0] == 1:
            self.yspeed -= 0.5
            self.xspeed -= 0.05
        if input[2] == 1:
            self.yspeed += 0.5
            self.xspeed += 0.05

        self.angle = round(math.atan2(self.yspeed, self.xspeed) / 0.017453292519943295)

        return self.reward, self.done, self.dist/(self.frames+40), self.inputs