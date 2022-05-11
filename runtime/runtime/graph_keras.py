from matplotlib import pyplot as plt

a = [24526.4253616333, 24399.231672286987, 24442.43281615819, 24437.33900843465, 24476.42224706577, 24475.43293650316, 24506.83685101479, 24524.50671936299, 24537.33167489083, 24451.54413778535]
b = [256303.6434650421, 249321.50077819824, 256547.8753627453, 249741.8396472931, 247546.98657989502, 255889.4952193886, 256818.8916057801, 253136.6511636468, 250562.2741344004, 259439.1410456818]


c =range(1,11)

figure, axis = plt.subplots(2, 2)


axis[0,0].scatter(c,a, c="r")
axis[0,0].set_title("Keras model coverted to TensorRT model")
axis[0,0].set_xlabel("Iterations")
axis[0,0].set_ylabel("Inference Time (ms)")
axis[0,0].set_ylim(min(a)*0.99,max(a)*1.01)

axis[0,1].scatter(c,b, c="r")
axis[0,1].set_title("Vanila Caffe model")
axis[0,1].set_xlabel("Iterations")
axis[0,1].set_ylabel("Inference Time (ms)")
axis[0,1].set_ylim(min(b)*0.99,max(b)*1.01)


d = ["Caffe", "TensorRT"]
e = [sum(a)/15570, sum(b)/15570]
print(sum(a)/15570)
print(sum(b)/15570)
axis[1,0].barh(d,e,height=0.1,align='center')
axis[1,0].set_title("Vanila Keras model")
axis[1,0].set_xlabel("Iterations")
axis[1,0].set_ylabel("Inference Time (ms)")
# axis[1,0].set_ylim(min(b)*0.99,max(b)*1.01)

plt.show()