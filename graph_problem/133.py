# 133 无向连通图中一个节点的引用，请你返回该图的深拷贝
# 每个节点都包含val 和 list[Node]
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

# 这个是2020年的代码
class Solution:
    def cloneGraph(self, node : 'Node') -> 'Node':
        lookup = {}
        def dfs(node):
            """
            输入是一个原始的node
            输出是克隆后的node
            """
            if not node: return
            if node in lookup:
                return lookup[node]
            clone = Node(node.val)
            lookup[node] = node
            for nbr in node.neighbors:
                clone.neighbors.append(dfs(nbr))
            return clone
        return dfs(node)


# 这个是2022年的代码
# 2020的代码字典的键是Node，而2022年的代码 键是结点的int
class Solution1:
    def cloneGraph(self, node : 'Node') -> 'Node':
        if not node: return None
        if not node.neighbors: return Node(1)
        q, ans = deque(), dict()
        visited = set()
        q.append(node)
        while q:
            node = q.popleft()
            if node.val in visited:
                continue
            if node.val not in ans:
                ans[node.val] = Node(node.val)
            for n in node.neighbors:
                if n.val not in ans:
                    ans[n.val] = Node(n.val)
                ans[node.val].neighbors.append(ans[n.val])
                if n.val not in visited:
                    q.append(n)
            visited.add(node.val)
        return ans[1]
