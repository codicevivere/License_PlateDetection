
import numpy as np
import cv2

# # Load an color image in grayscale
# img = cv2.imread('1.jpg',0)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# mu, sigma = 100, 15
# x = mu + sigma*np.random.randn(10000)

# # the histogram of the data
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# # add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=1)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)

# plt.show()

random_image = np.random.random([500, 500])
fig, plot = plt.subplots()
plot.imshow(random_image, cmap='gray', interpolation='nearest');
fig.show()
cv2.waitKey(0)
cv2.destroyAllWindows()