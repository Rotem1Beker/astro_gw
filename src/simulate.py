 # יצירת גלי GW + רעש
import numpy as np

#יוצר אות שתדירותו הולכת וגדלה, כמות אות מיזוג כוכבים שחורים
def generate_chirp(m1, m2, fs, duration):
    t = np.linspace(0, duration, int(fs * duration))
    f0 = 30
    k = (m1 + m2) * 5
    phase = 2 * np.pi * (f0 * t + k * t**2)
    h = np.sin(phase) * np.exp(-t)
    return h

#רעש גאוסי לבן
def add_noise(h, std):
    return h + np.random.normal(0, std, size=h.shape)

#דוגם פרמטרים פיזיקאלים אקראיים
def sample_parameters():
    m1 = np.random.uniform(10, 40)
    m2 = np.random.uniform(10, 40)
    return np.array([m1, m2])

#פונקציה מרכזית שמייצרת דוגמת אימון אחת
def simulate(fs, duration, noise_std):
    theta = sample_parameters()
    h = generate_chirp(theta[0], theta[1], fs, duration)
    x = add_noise(h, noise_std)
    return x, theta
