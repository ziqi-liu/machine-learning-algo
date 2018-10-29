# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import KDTree
class Node:
    def __init__(self, value = None, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right

def dis(a, b):
    return np.sum((a - b)**2)

class kdtree:
    def __init__(self, points, depth = 0):
        try:
            points = np.array(points)
            self.n, self.k = np.shape(points)
        except:
            self.tree = None
        else:
            if not self.n:
                self.tree = None
            else:
                axis = depth % self.k
                points = points[points[:,axis].argsort()]
                median = self.n // 2
                self.tree = Node(points[median], kdtree(points[:median], depth + 1).tree, kdtree(points[median + 1: ], depth + 1).tree)
        
    def inordertravesal(self, t):
        leftlist, rightlist = [], []
        if t.left:
            leftlist = self.inordertravesal(t.left)
        if t.right:
            rightlist = self.inordertravesal(t.right)
        return leftlist + [t.value] + rightlist
    
    def insert(self, point):
        try:
            point = np.asarray(point)
        except:
            raise ValueError
        if point.shape != tuple([self.k]):
            raise ValueError
        leaf = self.tree
        depth = 0
        while leaf:
            axis = depth % self.k
            if point[axis] < leaf.value[axis]:
                if not leaf.left:
                    leaf.left = Node(point, None, None)
                    break
                else:
                    leaf = leaf.left
            else:
                if not leaf.right:
                    leaf.right = Node(point, None, None)
                    break
                else:
                    leaf = leaf.right
            depth += 1
        self.n += 1
    
    def delete(self, point):
        #find the point
        point = np.asarray(point)
        prev = None
        prevleft = True
        target = self.tree
        depth = 0
        while target:
            axis = depth % self.k
            if point[axis] < target.value[axis]:
                prev = target
                target = target.left
            elif point[axis] == target.value[axis]:
                break
            else:
                prev = target
                prevleft = False
                target = target.right
            depth += 1
        else:
            raise ValueError('Point is not in the tree.')
        rebuild = target.inordertravesal()
        for i, e in enumerate(rebuild):
            if sum(abs(e - target.value)) < 1e-10:
                rebuild = rebuild[:i] + rebuild[i + 1:]
                break
        if not prev:
            self.tree = kdtree(rebuild).tree
        else:
            if prevleft:
                prev.left = kdtree(rebuild).tree
            else:
                prev.right = kdtree(rebuild).tree
        self.n -= 1
                
    def nn(self, point, t, depth = 0):
        path = [t]
        currentbestpoint = None
        currentbestdistance = float('inf')
        point = np.asarray(point)
        target = t
        tempdepth = depth
        prevtarget = None
        condition = 0
        while target.left or target.right:
            axis = tempdepth % self.k
            if point[axis] < target.value[axis]:
                if target.left:
                    target = target.left
                else:
                    condition = 1
                    break
            else:
                if target.right:
                    target = target.right
                else:
                    condition = 1
                    break
            tempdepth += 1
            path.append(target)
        if path and not condition:
            target = path.pop()
            prevtarget = target
            tempdepth -= 1
            currentbestpoint = target.value
            currentbestdistance = dis(point, target.value)
        while path:
            target = path.pop()
            if dis(point, target.value) < currentbestdistance:
                currentbestpoint = target.value
                currentbestdistance = dis(point, target.value)
            axis = tempdepth % self.k
            if (point[axis] - target.value[axis]) ** 2 < currentbestdistance:
                potentialbestpoint, potentialbestdistance = None, None
                if target.left and target.left != prevtarget :
                    potentialbestpoint, potentialbestdistance = self.nn(point, target.left, tempdepth + 1)
                elif target.right and target.right != prevtarget:
                    potentialbestpoint, potentialbestdistance = self.nn(point, target.right, tempdepth + 1)
                if potentialbestdistance and potentialbestdistance < currentbestdistance:
                    currentbestpoint, currentbestdistance = potentialbestpoint, potentialbestdistance
            prevtarget = target
            tempdepth -= 1
        return currentbestpoint, currentbestdistance
            
                
### Example

n = 1000
k = 4
data = np.random.normal(size = (n, k))
kd1 = KDTree(data)
kd2 = kdtree(data)
res1 = kd1.query([0, 0,0,0])
res2 = kd2.nn([0, 0,0,0], kd2.tree)
print(res1[0], np.sqrt(res2[1]))
print(kd1.data[res1[1]], res2[0])
