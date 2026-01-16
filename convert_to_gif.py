import imageio.v2 as imageio
import glob
import os

def main(task="vae"):
    files = glob.glob(os.path.join(f"./{task}", '**', '*.png'), recursive=True)
    img = []

    for file in files:
        img.append(imageio.imread(file))

    imageio.mimsave(f"./{task}.gif", img)

if __name__ == "__main__":
    # main("vae")
    main("dcgan")
