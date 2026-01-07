import h5py
import numpy as np

# קישור לקובץ strain של H1, 32 שניות
url = "https://losc.ligo.org/s/events/GW150914/H-H1_LOSC_4_V1-1126259446-32.hdf5"

# פותח זאת ישירות (לפעמים צריך קודם להוריד לקובץ)
print("Opening LIGO HDF5 data...")
f = h5py.File(url, "r")

# הדאטה
strain = np.array(f["/strain/Strain"])
gps_start = f["/meta/GPSstart"][()]
duration = f["/meta/Duration"][()]

print("shape:", strain.shape)
print("GPS start:", gps_start)
print("Duration (s):", duration)

# אפשר לשמור מקומית
with open("GW150914_H1_32s.npy", "wb") as out:
    np.save(out, strain)

print("Saved as GW150914_H1_32s.npy")
f.close()
