from PIL import Image

img = Image.open("gray_img.jpg") 

pixels = img.load()

for i in range(img.size[0]):
    for j in range(img.size[1]):
        x = pixels[i,j]
        x = abs(x-255)
        pixels[i,j] = x

img.save("conv_gray_img.jpg")
