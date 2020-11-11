import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    #x=(y-b)/m
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)  
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters) #first in array is slope 2nd is y intersept
        slope = parameters [0]
        intercept = parameters [1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # print(left_fit_average, 'left')
    # print(right_fit_average, 'right')
    try:
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    except Exception as e:
        print(e, '\n') #print error to console
        return None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_line(image, lines):
    line_image = np.zeros_like(image) #zero array, gambar hitam semua
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) #change to 1D array
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5) #draw line on top of black image
    return line_image

def region_of_interest(image):
    height, width = image.shape[0], image.shape[1]
    # width_centre = int(0.5*width)
    # height_centre = int(0.5*height)
    # polygons = np.array([
    #     [(0, int(0.9*height)), (int(0.8*width), height), (500, 300)]
    #     ])
    polygons = np.array([
        [(0, height), (width, height), (500, 300)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

image = cv2.imread('road-image/test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
croppped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(croppped_image, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_line(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow("result", combo_image)
cv2.waitKey(0)

# cap = cv2.VideoCapture("https://969e08f67492.ap.ngrok.io/blackvue_live.cgi")
# cap = cv2.VideoCapture("road-video/test2.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny(frame)
#     croppped_image = region_of_interest(canny_image)
#     lines = cv2.HoughLinesP(croppped_image, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercept(frame, lines) 
#     line_image = display_line(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

#     cv2.imshow("result", combo_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# https://www.youtube.com/watch?v=eLTLtUVuuy4&ab_channel=ProgrammingKnowledge

