from PIL import Image    
import glob
import os

frames = []
image_paths = glob.glob("pretrained/cdcgan/epoch*.png")
ordered = []
for path in image_paths:
    epoch = int(path.split(os.path.sep)[-1].split('_')[0][5:])
    ordered.append(epoch)
ordered.sort()
print(ordered)

for epoch_num in ordered:
    path = glob.glob(f"pretrained/cdcgan/epoch{epoch_num}*.png")[0]
    frame = Image.open(path)
    frames.append(frame)
    # print(path, epoch_num)

frames[0].save(os.path.join('animation.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
