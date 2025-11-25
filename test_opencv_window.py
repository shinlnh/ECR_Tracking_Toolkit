import cv2
import numpy as np

# Create a test image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 3)
cv2.putText(img, "Test Window", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Show window
print("Showing test window...")
print("Press any key to close")
cv2.imshow('Test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Window closed!")
