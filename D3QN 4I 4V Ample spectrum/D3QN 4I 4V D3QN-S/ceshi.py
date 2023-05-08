import numpy as  np

# V2V_Shadowing = np.random.normal(0, 3, [8, 8])
# V2V_channels_abs = np.zeros((8, 8))
# V2V_channels_abs = V2V_Shadowing
#
# V2V_channels_with_fastfading = np.repeat(V2V_channels_abs[:, :, np.newaxis], 4, axis=2)
# a = V2V_channels_with_fastfading[:, 1, :]
# b = V2V_channels_abs[:,0]
# print(a-b[:,None])

up_lanes = [i  for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i  for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i  for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i  for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]

print(up_lanes)
print(down_lanes)
print(left_lanes)
print(right_lanes)
