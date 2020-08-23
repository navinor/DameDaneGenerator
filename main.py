print('Importing...')
print('...Imageio')
import imageio
print('...Numpy')
import numpy as np
print('...Matplotlib')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
print('...Skimage')
from skimage.transform import resize
from skimage import img_as_ubyte
print('...Warnings')
import warnings
warnings.filterwarnings("ignore")
print('...Source Image and Driving Video')
source_image = imageio.imread('02.png')
driving_video = imageio.mimread('04.mp4')
print('...First order model API')
from firstordermodel.demo import load_checkpoints
from firstordermodel.demo import make_animation
print('...MoviePy (For speeding up the video and adding music)')
from moviepy.editor import VideoFileClip, vfx, AudioFileClip, CompositeAudioClip

#Resize image and video to 256x256
print('Resizing...')
print('...Source Image')
source_image = resize(source_image, (256, 256))[..., :3]
print('...Driving Video')
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]


def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source, driving[i]]
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani

print('Animating...')
print('...Loading Checkpoints')
generator, kp_detector = load_checkpoints(config_path='firstordermodel/config/vox-256.yaml',
                            checkpoint_path='firstordermodel/vox-cpk.pth.tar')
print('...Making Animation')
predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
print('Saving Video...')
imageio.mimsave('generated.mp4', [img_as_ubyte(frame) for frame in predictions])
print('...Speeding video up by 1.5x')
clip = VideoFileClip('generated.mp4')
finalclip = clip.fx(vfx.speedx, 3)

print('...Adding Background Music')
audio = AudioFileClip('audio.mp3')
print('...Exporting final clip...')
finalclip = finalclip.set_audio(audio)


finalclip.write_videofile('generated_with_sound.mp4')
print('Done! Your video has been exported! File name = "generated_with_sound.mp4')
