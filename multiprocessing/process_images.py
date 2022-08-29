import time
from PIL import Image, ImageFilter
import os, glob
import concurrent.futures

img_list = [img for img in glob.glob(os.getcwd() + '/*.jpg')]

t1 = time.perf_counter()
#t1 = time.time()

size = (1200, 1200)

#for img in img_list:
def process_image(img):
    img_name = img.split('/')[-1]
    img = Image.open(img)
    img = img.filter(ImageFilter.GaussianBlur(10))
    img = img.filter(ImageFilter.GaussianBlur(10))
    img.thumbnail(size)
    img.save(f'processed/{img_name}')
    print(f"{img_name} was processed")

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_image, img_list)

t2 = time.perf_counter()
#t2 = time.time()
print(f"Finished in {t2-t1} seconds")