import cv2
import gtk


window = gtk.gdk.devices_list()
print(window)
# shape = window.get_size()

# print("The size of the window is %d x %d" % shape)

# pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, shape[0], shape[1])
# pb = pb.get_from_drawable(window, window.get_colormap(),
#                           0, 0, 0, 0, shape[0], shape[1])

# if (pb is not None):
#     print(pb.save("1.png", "png"))
#     print("Screenshot saved to screenshot.png.")
# else:
#     print("Unable to get the screenshot.")

# src = cv2.imread("1.png")
# cv2.imshow("src", src)
# cv2.waitKey(0)
