import  numpy as np

# z = []
# x = []
# for index in range(5):
#     z.append([])
#     for j in range(3):
#         z[-1].append(j)
#         desc = j
#
# print(z)

def testDict():
    dic = {}
    for i in range(10):
        dic[i] = "abc"+str(i)

    index = 5
    print(dic)
    print(dic[index])


testDict()