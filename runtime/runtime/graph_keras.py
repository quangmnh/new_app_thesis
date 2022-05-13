from matplotlib import pyplot as plt

f = [24526.4253616333, 24399.231672286987, 24442.43281615819, 24437.33900843465, 24476.42224706577, 24475.43293650316, 24506.83685101479, 24524.50671936299, 24537.33167489083, 24451.54413778535]
b = [15860.400438308716, 16003.870643178016, 15925.209657683425, 15887.1520994536, 15846.181471376374, 15825.229221967473, 16007.650881747957, 15798.100303473751, 15961.959201080297, 15780.764892232613]
a = []
for i in f:
    a.append(i/1554)

c =range(1,11)

# figure, axis = plt.subplots(2, 2)


# plt.scatter(c,a, c="r")
# plt.title("Keras model coverted to TensorRT model\n")
# plt.xlabel("Iterations")
# plt.ylabel("Inference Time (ms)")
# plt.ylim(min(a)*0.99,max(a)*1.01)

# plt.scatter(c,b, c="r")
# plt.title("Vanila Keras model")
# plt.xlabel("Iterations")
# plt.ylabel("Inference Time (ms)")
# plt.ylim(min(b)*0.99,max(b)*1.01)


d = ["TensorRT", "Keras"]
e = [sum(a)/15540, sum(b)]
plt.barh(d,e,height=0.1,align='center')
plt.title("Comparing the 2 model\n Average inference time")
plt.ylabel("Model type")
plt.xlabel("Inference Time (ms)")

plt.show()