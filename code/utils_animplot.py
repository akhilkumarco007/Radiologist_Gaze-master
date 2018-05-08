import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from config import args


def gazePlotter(data, name=None, flattened=True, save=False, output_path=None, plot=True):
    '''
    Plots and saves the gaze pattern as animation
    :param data: gaze data vector
    :param name: name of the file to be saved
    :param flattened: boolean to know if the data vector is flat
    :param save: boolean to determine save or not
    :param output_path: string path to the output vector
    :param plot: Boolean to display the plot
    :return: none
    '''
    x = []
    y = []
    if flattened:
        for i in range(0, len(data) - 1, 2):
            x.append(data[i])
            y.append(data[i + 1])
    else:
        for point in data:
            x.append(point[0])
            y.append(point[1])
    fig = plt.figure()
    # plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/ffmpeg.exe'
    plt.xlim(0, 2560)
    plt.ylim(1440, 0)
    line, = plt.plot(x, y)

    def animate(ii):
        line.set_data(x[:ii + 1], y[:ii + 1])
        return line

    writer = animation.ImageMagickFileWriter()  # for ubuntu
    # writers = animation.writers['ffmpeg']
    # writer = writers(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # writer = animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    img = plt.imread(args.proj_dir + '/images/CXR129_IM-0189-1001.jpg')
    plt.imshow(img)
    ani = animation.FuncAnimation(fig, animate, frames=50, interval=100)
    if save is True:
        if not os.path.exists(output_path + str(name.split('/')[0])):
            os.makedirs(output_path + str(name.split('/')[0]))
        ani.save(output_path + str(name) + str() + '.mp4', writer='ffmpeg', fps=15)
    if plot is True:
        plt.show()


def gaze_plot_save(cluster_sorted, imp_centers, num=5, path='carol_most_important/'):
    '''
    saves the plot as an animation in the given path
    :param cluster_sorted: Array of clusters that are sorted based on importance from the classifier
    :param imp_centers: Array of the important centers vector from the classifier
    :param num: Scalar quantity determines the number of plots to be saved from the most important vectors
    :param path: output save path
    :return: None
    '''
    for i in range(1, num+1):
        gazePlotter(imp_centers[cluster_sorted[-i][0]], name=path + str(i) + '_' + str(cluster_sorted[-i][1]),
                    flattened=True, save=True, output_path=args.path_to_videos, plot=False)
        print('Creating and Saving Animation.....')


def rawSubPlotter(filteredGaze, imgName, output_path, save=False, plot=True, name=None):
    '''
    Helper function to plot gaze data after filtering and warping
    '''
    gaze_ = filteredGaze[imgName]
    gazePlotter(gaze_, flattened=False, save=save, output_path=output_path, plot=plot, name=name)
