from email.mime import image
from PIL import Image
from pytesseract import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import pandas as pd

# Defining paths to tesseract.exe
# and the image we would be using
image_path = "8.png"
image_path_new = "cv0.png"
  
# Opening the image & storing it in an image object
# img = Image.opIen(image_path).convert('L')
img_old = cv2.imread(image_path)
img_old = cv2.resize(img_old, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)
# img.save('pil-greyscale.png')
cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imwrite(image_path_new, img)





# Providing the tesseract executable
# location to pytesseract library
# pytesseract.tesseract_cmd = path_to_tesseract
  
# Passing the image object to image_to_string() function
# This function will extract the text from the image
d = pytesseract.image_to_data(img, output_type=Output.DICT)

"""
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    print(d["text"][i])
    cv1.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv1.imwrite(image_path_new, img)
"""

# Look for hello world and goodbye world
n_boxes = len(d['level'])
hello = []
goodbye = []
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    text = (d["text"][i])
    if "Hello" in text and "print" in text:
        print("Found hello", x, y, w, h)
        hello = [x, y, w, h]
    if "Goodbye" in text and "print" in text:
        print("Found goodbye", x, y, w, h)
        goodbye = [x, y, w, h]

if len(hello) == 0:
    print("Could not find hello")
if len(goodbye) == 0:
    print("Could not find goodbye")

pad = 10
img_cropped = img[hello[1]-pad:goodbye[1]+goodbye[3]+pad, min(hello[0], goodbye[0])-pad:]

"""
d = pytesseract.image_to_data(img_cropped, config='-c preserve_interword_spaces=3 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    print(d)
    break
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    print(d["text"][i], w, h)
    cv2.rectangle(img_cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite(image_path_new, img_cropped)

e = pytesseract.image_to_string(img_cropped, config='-c preserve_interword_spaces=1')
# print(e)
"""

cv2.imwrite(image_path_new, img)
image_path_new = "cv1.png"
cv2.imwrite(image_path_new, img_cropped)

# Dialate 
kernel = np.ones((2,2), np.uint8)
img_cropped = cv2.dilate(img_cropped, kernel, iterations=2)
cv2.imwrite(image_path_new, img_cropped)

# Run OCR on the cropped image 
custom_config = r' -l eng --oem 1 --psm 6  -c preserve_interword_spaces=1 -c tessedit_char_whitelist="#ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_#%^&*()-_0123456789- "'
gauss = cv2.GaussianBlur(img_cropped, (3, 3), 0)
d = pytesseract.image_to_data(gauss, config=custom_config, output_type=Output.DICT)
df = pd.DataFrame(d)

# clean up blanks
df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]

# sort blocks vertically
sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
for block in sorted_blocks:
    curr = df1[df1['block_num'] == block]
    sel = curr[curr.text.str.len() > 3]
    char_w = (sel.width / sel.text.str.len()).mean()
    prev_par, prev_line, prev_left = 0, 0, 0
    text = ''
    for ix, ln in curr.iterrows():
        # add new line when necessary
        if prev_par != ln['par_num']:
            text += '\n'
            prev_par = ln['par_num']
            prev_line = ln['line_num']
            prev_left = 0
        elif prev_line != ln['line_num']:
            text += '\n'
            prev_line = ln['line_num']
            prev_left = 0

        added = 0  # num of spaces that should be added
        if ln['left'] / char_w > prev_left + 1:
            added = int((ln['left']) / char_w) - prev_left
            text += ' ' * added
        text += ln['text'] + ' '
        prev_left += len(ln['text']) + added + 1
    text += '\n'
    print(text)
"""
"""
cv2.imwrite(image_path_new, gauss)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    print(d["text"][i])
    cv2.rectangle(gauss, (x, y), (x + w, y + h), (0, 0, 0), 2)
# cv2.imwrite(image_path_new, img)