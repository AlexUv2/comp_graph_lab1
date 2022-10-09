import numpy as np

ax_len = 1500

axis_arr = np.array(
    [[0,    0,   1],
     [ax_len, 0, 1],
     [0, ax_len, 1]]
)

identity_matrix = np.identity(3, dtype=np.int32)
# identity_matrix = np.array(
#     [[2, 2, 0],
#      [0, 3, 0],
#      [0, 0, 1]]
# )

print()