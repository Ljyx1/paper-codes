import numpy as np

# V2V_Shadowing = np.random.normal(0, 3, [8, 8])
# V2V_channels_abs = np.zeros((8, 8))
# V2V_channels_abs = V2V_Shadowing
#
# V2V_channels_with_fastfading = np.repeat(V2V_channels_abs[:, :, np.newaxis], 4, axis=2)
# a = V2V_channels_with_fastfading[:, 1, :]
# b = V2V_channels_abs[:,0]
# print(a-b[:,None])、
x = np.ones((4,4,2))
print(np.sum(x))
print(1446/60)

