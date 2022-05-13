from matplotlib import pyplot as plt

a = [28346.26340866089, 28408.08868408203, 28422.67314924785, 28366.67872214016, 28386.4611393574, 28343.18676076686, 28325.18460088126, 28407.54472347473, 28438.78745586606, 28439.75975982661]
b = [256303.6434650421, 249321.50077819824, 256547.8753627453, 249741.8396472931, 247546.98657989502, 255889.4952193886, 256818.8916057801, 253136.6511636468, 250562.2741344004, 259439.1410456818]


c =range(1,11)

# figure, axis = plt.subplots(3, 1)


# plt.scatter(c,a, c="r")
# plt.title("Caffe model coverted to TensorRT model\n 1557 samples each iteration")
# plt.xlabel("Iterations")
# plt.ylabel("Inference Time (ms)")
# plt.ylim(min(a)*0.99,max(a)*1.01)

# plt.scatter(c,b, c="r")
# plt.title("Vanila Caffe model \n 1557 samples each iteration")
# plt.xlabel("Iterations")
# plt.ylabel("Inference Time (ms)")
# plt.ylim(min(b)*0.99,max(b)*1.01)


d = ["Caffe", "TensorRT"]
e = [sum(a)/15570, sum(b)/15570]
print(sum(a)/15570)
print(sum(b)/15570)
plt.barh(d,e,height=0.1,align='center')
plt.title("Comparing the 2 model\n Average inference time")
plt.ylabel("Model type")
plt.xlabel("Inference Time (ms)")

plt.show()