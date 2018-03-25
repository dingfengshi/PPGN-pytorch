import PIL.Image as Image


def get_img_list(path):
    img_list = []
    with open(path) as fin:
        for eachline in fin.readlines():
            path = eachline.rstrip().split()[1]
            img_list.append(path[:-4])

    return img_list


def cut_pic():
    images_list_path = "/home/ec2-user/CUB_200_2011/images.txt"
    bound_boxes_path = "/home/ec2-user/CUB_200_2011/bounding_boxes.txt"
    images_path = "/home/ec2-user/CUB_200_2011/images/"
    img_list = get_img_list(images_list_path)
    idx = 0
    with open(bound_boxes_path) as fin:
        for each in fin.readlines():
            elem = each.rstrip().split()
            path = images_path + img_list[idx] + ".jpg"
            save_path = images_path + img_list[idx] + ".jpg"
            idx = idx + 1
            img = Image.open(path)
            x = int(float(elem[1]))
            y = int(float(elem[2]))
            width = int(float(elem[3]))
            height = int(float(elem[4]))

            x_mid = int(x + width / 2)
            y_mid = int(y + height / 2)
            img_x_bd = img.width
            img_y_bd = img.height

            if width < height:
                if x_mid - height / 2 >= 0 and x_mid + height / 2 < img_x_bd:
                    # 有足够位置增加x边框长度
                    width = height
                    x = x_mid - height / 2
                elif img_x_bd < height:
                    # 不够位置，取原图的x长度
                    x = 0
                    width = img_x_bd
                else:
                    # 够位置，但是不是居中
                    if x_mid - height / 2 < 0:
                        x = 0
                        width = height
                    else:
                        x = img_x_bd - height
                        width = height
            elif width > height:
                if y_mid - width / 2 >= 0 and y_mid + width / 2 < img_y_bd:
                    # 有足够位置增加y边框长度
                    height = width
                    y = y_mid - width / 2
                elif img_y_bd < width:
                    # 不够位置，取原图的y长度
                    y = 0
                    height = img_y_bd
                else:
                    # 够位置，但是不是居中
                    if y_mid - width / 2 < 0:
                        y = 0
                        height = width
                    else:
                        y = img_y_bd - width
                        height = width

            img = img.crop((x, y, x + width, y + height))
            img.save(save_path)


def delete_not_rgb_picture():
    images_list_path = "CUB/images.txt"
    images_path = "CUB/images/"
    img_list = get_img_list(images_list_path)

    for value in img_list:
        img = Image.open(images_path + value + ".jpg")
        if img.mode != 'RGB':
            print(value + "mode=" + img.mode)


if __name__ == '__main__':
    delete_not_rgb_picture()
