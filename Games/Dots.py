from skimage.transform import resize
import numpy as np
import itertools

'''
    I cloned open-source and modified a little
    Plz refer original version below site !!
    https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py
    
    written by hanseul-jeong
    https://github.com/hanseul-jeong/RL
'''

class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class gameEnv():
    def __init__(self, size, partial=False):
        '''
        Set Game environment
        :param size: Grid size (e.g., 4x4, 10x10)
        :param partial:
        '''

        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.GOOD = 5
        self.BAD = -5
        self.objects = []
        self.partial = partial
        a = self.reset()

    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition(), 1, 1, 1, self.GOOD, 'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition(), 1, 1, 0, self.BAD, 'fire')
        self.objects.append(hole)
        bug2 = gameOb(self.newPosition(), 1, 1, 1, self.GOOD, 'goal')
        self.objects.append(bug2)
        hole2 = gameOb(self.newPosition(), 1, 1, 0, self.BAD, 'fire')
        self.objects.append(hole2)
        bug3 = gameOb(self.newPosition(), 1, 1, 1, self.GOOD, 'goal')
        self.objects.append(bug3)
        bug4 = gameOb(self.newPosition(), 1, 1, 1, self.GOOD, 'goal')
        self.objects.append(bug4)
        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        done = False
        hero = self.objects[0]
        d_x = 0
        d_y = 0
        penalize = -0.5
        if direction == 0:
            d_y -= 1
        if direction == 1:
            d_y += 1
        if direction == 2:
            d_x -= 1
        if direction == 3:
            d_x += 1

        if hero.x + d_x in [-1, self.sizeX] or hero.y + d_y in [-1, self.sizeY]:
            penalize = self.BAD
            d_x = 0
            d_y = 0

        self.objects[0].x += d_x
        self.objects[0].y += d_y
        return penalize

    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == self.GOOD:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 1, self.GOOD, 'goal'))
                else:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 0, self.BAD, 'fire'))
                return other.reward, False
        if ended == False:
            return 0.0, False

    def renderEnv(self):
        # 2 for width and height padding
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        a[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y + 3, hero.x:hero.x + 3, :]

        b = resize(a[:, :, 0], [self.sizeX+2, self.sizeY+2])
        c = resize(a[:, :, 1], [self.sizeX+2, self.sizeY+2])
        d = resize(a[:, :, 2], [self.sizeX+2, self.sizeY+2])
        a = np.stack([b, c, d], axis=2)
        return a

    def step(self, action):
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state, (reward + penalty), done, None
        else:
            return state, (reward + penalty), done, None

    def find_optimal_action(self):
        assert self.objects[0].name == 'hero'
        current_pos = (self.objects[0].x, self.objects[0].y)
        others = self.objects[1:]
        actions = [(0,-1),(0,+1),(-1,0),(+1,0)]
        rewards = []
        for action in actions:
            rewards.append(0)
            cand = tuple(min(max(c+a, 0),self.sizeX-1) for c, a in zip(current_pos,action))
            for other in others:
                if (other.x, other.y) == cand:
                    rewards[-1] += other.reward
                    break
        max_action = np.argmax(rewards)
        return max_action, rewards[max_action]
