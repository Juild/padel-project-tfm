
import cv2
for idx in range(500, 600):
    # Load the image
    img = cv2.imread(f"../datasets/play_reference_frames/frame{idx}.jpg")


    # Define the mask
    mask = cv2.inRange(img, (100, 204, 160), (240, 255, 232))
    mask = cv2.inRange(img, (70, 100, 100), (255, 255, 250))
    mask = cv2.inRange(img, (110, 50, 50), (130, 255, 255))
    mask = cv2.inRange(img, (0, 0, 0), (255, 255, 255))
    mask = cv2.inRange(img, (200, 100, 100), (255, 255, 255))

    mask = cv2.inRange(img, (10, 200, 100), (255, 255, 255))





    # Apply the mask to the image
    res = cv2.bitwise_and(img, img, mask=mask)

    # Show the original and filtered image
    cv2.imshow("Original", img)
    cv2.imwrite(f'./mask_test_images/masked_frame{idx}.jpg', res)
# cv2.imshow("Green Filtered", res)

# Wait for a key press
# cv2.waitKey(0)

# Close all windows
# cv2.destroyAllWindows()


