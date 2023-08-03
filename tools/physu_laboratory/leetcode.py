class Solution:
    def letterCombinations(self, digits):
        dic = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv',
               '9': 'wxyz'}  # 首先定义一个哈希表，把相应的数字和字母对应起来
        n = len(digits)  # 计算给定字符串中数字的长度
        if n == 0:
            return []

        def dfs(index):  # 定义递归回溯函数
            if index == n:  # 如果此时递归到了字符串中最后一个数字，则表示第一层深度搜索结束，将结果存储起来
                res.append(''.join(tmp))  # 注意一下这边存储的时候要把多个字符串组合成一个字符串，利用
                # join函数，这边还要注意的是后面会把tmp中的最上面字符弹出，这边利用join函数也同时避免
                # 了tmp弹出时影响res的目的，不然全部返回空字符串
            else:  # 如果这个时候没搜索到最后一层，则
                for i in dic[digits[index]]:  # 逐个遍历当前数字对应的字母
                    tmp.append(i)  # 将这个字母加到tmp中
                    dfs(index + 1)  # 因为要把下一个数字对应的字母继续加进去，所以这边index + 1，函数传递下一个位置
                    tmp.pop()  # 因为还需要把最后一个数字的字母逐个加进去，所以这边要把最后一个弹出，
                    # 因为上一行的dfs函数遇到结束条件，才回向下执行，所以这边已经完成一次深度的搜索，此时要把最后一个字母弹出，实现下一轮遍历

        res = []  # 初始化返回结果数组
        tmp = []  # 初始化过渡数组
        dfs(0)  # 首先执行index = 0，因为都是从第一个字母开始组合
        return res  # 返回结果函数

a = Solution()
b = a.letterCombinations('345')
print(f'b:{b}')