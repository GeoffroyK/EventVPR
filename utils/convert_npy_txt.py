import numpy as np

traverses = ['sunset1', 'sunset2', 'morning', 'sunrise', 'daytime']
places = 25
timewindow = 1.0


for traverse in traverses:
    print(f'Converting {traverse}')
    for place in range(places):
        print(place)
        npy_arr = np.load(f"data/timewindow/{traverse}_{place}_{timewindow}.npy", allow_pickle=True)
        np.savetxt(f'./data/timewindow/{traverse}_{place}_{timewindow}.txt', npy_arr)
print("Done !")