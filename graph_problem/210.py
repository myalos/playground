# 课程表II
# 总共有numCourses门课要选，0~numCourses - 1
# prerequisites 先修课程 prerequisities = [ai, bi]
# 表示选ai前要修完bi
# 返回学完所有课程所安排的学习顺序，如果有多种，返回任意一种，如果没有返回一个空数组

from typing import *

class Solution:
    def findOrder(self, numCourses : int, prerequisities : List[List[int]]) -> List[int]:
        pass

def main():
    sol = Solution()
    _input = (2, [[1, 0]])
    _output = sol.findOrder(*_input)
    print(_output)

if __name__ == '__main__':
    main()
