import header as h
import numpy as np
import matplotlib

cell    = h.f_rdamr(171,100)
d   = h.d_gasmap(171, 100, cell, amrtype='den')
i   = h.g_img(d)

matplotlib.pyplot.imshow(i)
matplotlib.pyplot.show()
