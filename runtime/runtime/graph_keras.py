from matplotlib import pyplot as plt

a = [5096.033573150635, 5028.115588388263, 5188.541350908367, 5031.151940463104, 5027.166984252482, 5180.261882229292, 5029.117975579096, 5105.194769272201, 5042.087670181283, 5086.290595646324]
b = [344314.96596336365, 351855.140209198, 351871.97204468725, 348931.5975860581, 345376.1910304965, 346759.1915006889, 350542.4948814845, 348764.7173796807, 344505.7678921289, 348419.72281439696]


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
e = [sum(a)/15540, sum(b)/15540]
plt.barh(d,e,height=0.1,align='center')
plt.title("Comparing the 2 model\n Average inference time")
plt.ylabel("Model type")
plt.xlabel("Inference Time (ms)")

plt.show()