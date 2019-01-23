import re
from PIL import Image
from PIL import ImageEnhance
import glob as gb
import os

original_dir = os.path.expanduser("~/Disk/ic-data/clean_train_data")
dst_dir = os.path.expanduser("~/Disk/ic-data/color_data/")
for file in os.listdir(original_dir):
    result = re.findall(r"(.*).jpg", file)
    number = result[0]

    path = os.path.join(original_dir, file)
    im = Image.open(path)

    # 亮度增强
    for i in [2,4]:
        brightness = 1 + 0.5 * i
        dst_bri_path = dst_dir + result[0] + '_bright' + str(brightness) + '.jpg'
        enh_bri = ImageEnhance.Brightness(im)
        image_brightened = enh_bri.enhance(brightness)
        image_brightened.save(dst_bri_path, 'JPEG')
    # dst_flip_bri_path = 'image_data/' + result[0] + '_flip_bright' + str(brightness) + '.jpg'
    #     enh_bri = ImageEnhance.Brightness(dst_flip)
    #     image_brightened = enh_bri.enhance(brightness)
    #     image_brightened.save(dst_flip_bri_path, 'JPEG')

    # 色度增强
    for i in [2,4]:
        color = 1 + 0.5 * i
        dst_col_path = dst_dir + result[0] + '_color' + str(color) + '.jpg'
        enh_col = ImageEnhance.Color(im)
        image_colored = enh_col.enhance(color)
        image_colored.save(dst_col_path, 'JPEG')
    #
    # 对比度增强
    for i in [2,4]:
        contrast = 1 + 0.5 * i
        dst_con_path = dst_dir + result[0] + '_contrast' + str(contrast) + '.jpg'
        enh_con = ImageEnhance.Contrast(im)
        image_contrasted = enh_con.enhance(contrast)
        image_contrasted.save(dst_con_path, 'JPEG')
    #
    # 锐度增强
    for i in [2,4]:
        sharpness = i
        dst_sha_path = dst_dir + result[0] + '_sharp' + str(sharpness) + '.jpg'
        enh_sha = ImageEnhance.Sharpness(im)
        image_sharped = enh_sha.enhance(sharpness)
        image_sharped.save(dst_sha_path, 'JPEG')