from PIL import Image, ImageFilter, ImageEnhance
import pandas as pd
import os
import random

RAW_IMAGE_PATH = "./input/train_images/"
RESIZED_IMAGE_PATH = "./input/resize_images/"
AUGMENTED_IMAGE_PATH = "./input/augmented_images/"
train_csv= pd.read_csv("./input/train.csv")
train_csv["label"] = train_csv["label"].astype(int)

TARGET_SIZE = (32, 32)

if __name__ == "__main__":
    list_dir=os.listdir(RAW_IMAGE_PATH)
    for file in list_dir:
        img = Image.open(RAW_IMAGE_PATH + file)
        img = img.resize(TARGET_SIZE)
        print(RESIZED_IMAGE_PATH + file)
        img.save(RESIZED_IMAGE_PATH + file)

    new_csv = pd.DataFrame(columns = ["image_id","label"])
    print(new_csv.head())
    for i in range(2):
        for row in train_csv.index:
            img = Image.open(RAW_IMAGE_PATH + train_csv["image_id"][row])
            img = img.convert("RGB")

            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0,3.0)))
            img = img.rotate(random.randint(-70,70))
            img = img.transpose(method=random.choice([Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))

            width, height = img.size   # Get dimensions
            left = width/4
            top = height/4
            right = 3 * width/4
            bottom = 3 * height/4
            img = img.crop((left, top, right, bottom))

            img = img.resize(TARGET_SIZE)

            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.5, 1.5)
            im_output = enhancer.enhance(factor)
            im_output.save(AUGMENTED_IMAGE_PATH + str(i) + "_" + train_csv["image_id"][row])

            new_row={"image_id": str(i) + "_" + train_csv["image_id"][row],"label": train_csv["label"][row]}
            new_csv=new_csv.append(new_row, ignore_index=True)
    new_csv.to_csv(r'./input/train_augmented.csv',index=False)