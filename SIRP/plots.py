import numpy as np
import matplotlib.pyplot as plt


def circular_aperture(image, aperture_radius, position=None, inverted=False):
    """ function for a circular aperture

    :param image: image
    :type image: np.ndarray
    :param aperture_radius: radius of the aperture
    :type aperture_radius: float
    :param position: center y and x coordinates of the aperture
    :type position: list or np.ndarray
    :param inverted: if the mask is inverted or not
    :type inverted: bool
    :return: mask
    :rtype: np.ndarray
    """
    mask = np.ones(image.shape)
    y_length = image.shape[1]
    x_length = image.shape[0]

    if position is None:
        position = (int(y_length / 2), int(x_length / 2))

    yy, xx = np.ogrid[:x_length, :y_length]

    rad = np.sqrt((xx - position[1]) ** 2 + (yy - position[0]) ** 2)

    if inverted:
        mask[(rad < aperture_radius)] = 0
    else:
        mask[(rad > aperture_radius)] = 0

    return mask


def get_corner(angle):
    """ function to obtain the corner in which the two orientation vectors fit best.

    :param angle: angle of orientation
    :type angle: float, int
    :return: name of the corner
    :rtype: str
    """

    if angle < 45:
        corner = 'bottom right'
    elif 45 <= angle < 135:
        corner = 'top right'
    elif 135 <= angle < 225:
        corner = 'top left'
    elif 225 <= angle < 315:
        corner = 'bottom left'
    elif 315 <= angle:
        corner = 'bottom right'
    else:
        raise ValueError('wrong angle!')
    return corner

def plot_pixel_scale(axs, pixel_scale, orientation, bar_length=None, color='white', parallax=None):
    """ function to plot the pixel scale of the image

    :param axs: x- and y-axis of the plot, can be obtained with plt.gca()
    :type axs: matplotlib.pyplot.axes
    :param pixel_scale: pixel scale of the detector in arc-seconds/pixel
    :type pixel_scale: float, int
    :param orientation: rotation angle of the images, need as input as if it is not none WCS is plotted and
     pixel scale should be opposite corner
    :type orientation: float, int
    :return: None
    :rtype: None
    """

    if orientation is None:
        corner = "bottom right"
    else:
        corner = get_corner(orientation + 180)

    scale_factor = pixel_scale  # arcsec/pixel

    y_lim = axs.get_ylim()
    x_lim = axs.get_xlim()

    x_length = x_lim[1] - x_lim[0]
    y_length = y_lim[1] - y_lim[0]

    length = np.mean([x_length, y_length])
    quarter_image_arc_sec = length / 3 * scale_factor

    if bar_length is None:
        bar_length = (
                                 quarter_image_arc_sec // 5) * 5  # divide by 5 with rest and then multipy by 5 -> final lengths in  steps of 5 arcsecs

    line_length = bar_length / scale_factor  # length of line in pixels

    if corner == 'top left':
        start_x = x_lim[0] + 0.05 * length
        start_y = y_lim[0] + 0.95 * length
        text_y = y_lim[0] + 0.90 * length
    elif corner == 'top right':
        start_x = x_lim[0] + 0.95 * length - line_length
        start_y = y_lim[0] + 0.95 * length
        text_y = y_lim[0] + 0.90 * length
    elif corner == 'bottom right':
        start_x = x_lim[0] + 0.95 * length - line_length
        start_y = y_lim[0] + 0.05 * length
        text_y = y_lim[0] + 0.07 * length
    elif corner == 'bottom left':
        start_x = x_lim[0] + 0.08 * length
        start_y = y_lim[0] + 0.05 * length
        text_y = y_lim[0] + 0.07 * length
    else:
        raise ValueError('false corner')
    

    lims = axs.get_xlim()
    xlength = abs(lims[1] - lims[0])
    line_length_ratio = line_length / xlength
    line_length = line_length_ratio

    if corner == 'top left':
        start_x = 0.02
        start_y = 0.98
        text_y = 0.96
    elif corner == 'top right':
        start_x = 0.98 - line_length
        start_y = 0.98
        text_y = 0.96
    elif corner == 'bottom right':
        start_x = 0.98 - line_length
        start_y = 0.02
        text_y = 0.04
    elif corner == 'bottom left':
        start_x = 0.02
        start_y = 0.02
        text_y = 0.04
    else:
        raise ValueError('false corner')

    axs.hlines(start_y, start_x, start_x + line_length, color=color, linewidth=2, transform=axs.transAxes)
    
    if bar_length >= 10:
        if bar_length / parallax > 1:
            axs.text(start_x, text_y, f'{bar_length:.0f}" = {bar_length / parallax:3.0f} AU', color=color, va='center',
                     ha='left', fontsize=14, transform=axs.transAxes)
        else:
            axs.text(start_x, text_y, f'{bar_length:.0f}" = {bar_length / parallax:1.3f} AU', color=color, va='center',
                     ha='left', fontsize=14, transform=axs.transAxes)

    elif bar_length >= 1:
        if bar_length / parallax > 1:
            axs.text(start_x, text_y, f'{bar_length:.1f}" = {bar_length / parallax:3.0f} AU', color=color, va='center',
                     ha='left', fontsize=14, transform=axs.transAxes)
        else:
            axs.text(start_x, text_y, f'{bar_length:.1f}" = {bar_length / parallax:1.3f} AU', color=color, va='center',
                     ha='left', fontsize=14, transform=axs.transAxes)

    elif bar_length < 1:
        if bar_length / parallax > 1:
            axs.text(start_x, text_y, f'{bar_length * 1000:3.1f} mas = {bar_length / parallax:3.0f} AU', color=color,
                     va='center', ha='left', fontsize=14, transform=axs.transAxes)
        else:
            axs.text(start_x, text_y, f'{bar_length * 1000:3.1f} mas = {bar_length / parallax:1.3f} AU', color=color,
                     va='center', ha='left', fontsize=14, transform=axs.transAxes)

    return


def plot_orientation(axs, orientation, corner=None, color='white'):
    """ function to plot the WCS coordinate system.

    :param axs: x- and y-axis of the plot, can be obtained with plt.gca()
    :type axs: matplotlib.pyplot.axes
    :param orientation: rotation angle of the images to orientate the WCS coordinate system
    :type orientation: float, int
    :return: None
    :rtype: None
    """

    orientation = orientation % 360

    y_lim = axs.get_ylim()
    x_lim = axs.get_xlim()

    x_length = x_lim[1] - x_lim[0]
    y_length = y_lim[1] - y_lim[0]

    length = np.mean([x_length, y_length])

    sin = np.sin(orientation / 180 * np.pi)
    cos = np.cos(orientation / 180 * np.pi)

    N_vector = np.array([-length * 0.1 * sin, length * 0.1 * cos])
    E_vector = np.array([-length * 0.1 * cos, - length * 0.1 * sin])

    if corner is None:
        corner = get_corner(orientation)

    if corner == 'top left':
        start_x = x_lim[0] + 0.05 * length - np.min([N_vector[0], 0])
        start_y = y_lim[0] + 0.95 * length - np.max([E_vector[1], 0])
    elif corner == 'top right':
        start_x = x_lim[0] + 0.95 * length - np.max([E_vector[0], 0])
        start_y = y_lim[0] + 0.95 * length - np.max([N_vector[1], 0])
    elif corner == 'bottom right':
        start_x = x_lim[0] + 0.95 * length - np.max([N_vector[0], 0])
        start_y = y_lim[0] + 0.05 * length - np.min([E_vector[1], 0])
    elif corner == 'bottom left':
        start_x = x_lim[0] + 0.1 * length - np.min([E_vector[0], 0])
        start_y = y_lim[0] + 0.05 * length - np.min([N_vector[1], 0])
    else:
        raise ValueError('false corner')

    # y axis
    axs.arrow(start_x, start_y, N_vector[0], N_vector[1], color=color, head_width=length * 0.02, width=1.2)
    axs.text(start_x + N_vector[0] * 1.7, start_y + N_vector[1] * 1.7, 'N', color=color, va='center',
             ha='center', fontsize=16)
    # x axis
    axs.arrow(start_x, start_y, E_vector[0], E_vector[1], color=color, head_width=length * 0.02, width=1.2)
    axs.text(start_x + E_vector[0] * 1.7, start_y + E_vector[1] * 1.7, 'E', color=color, va='center',
             ha='center', fontsize=16)

    return


def show_img(img, xlim=None, ylim=None, minper=1, maxper=99, show_center=False, orientation=None, size=5,
             pixel_scale=None, cmap='plasma', ticks=True, norm=None, mask_center_rad=None, title = None, dpi=200, vmin=None,vmax=None):
    """ plot an HCI image

    :param img: data of the image(s)
    :type img: np.ndarray or list of np.ndarray
    :param xlim: x-axis limits for the plots
    :type xlim: list or np.ndarray
    :param ylim: x-axis limits for the plots
    :type ylim: list or np.ndarray
    :param minper: minimum percentage for the shown intensity, default is 1%
    :type minper: float or int
    :param maxper: maximum percentage for the shown intensity, default is 99%
    :type maxper: float or int
    :param show_center: weather to plot a red cross in the center or not, default is False
    :type show_center: bool
    :param orientation: rotation angle of the images, need as input as if it is not none WCS is plotted and
     pixel scale should be opposite corner
    :type orientation: float, int
    :param size: size in inches of the plot, default is 3 by 3 inches
    :type size: int
    :param pixel_scale: pixel scale of the detector in arc-seconds/pixel
    :type pixel_scale: float, int
    :param cmap: matplotlib colormap
    :type cmap: str
    :param ticks: weather or not to plot the x and y ticks
    :type ticks: bool
    :return: None
    :rtype: None
    """

    if isinstance(img, list):
        figsize = [size * len(img), size]
        
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            ratio = img.shape[0]/img.shape[1]
            img = [img]
            title = [title]

            figsize = [size, ratio*size] 
        elif len(img.shape) == 3:
            figsize = [size * len(img), size]


    if norm is not None and ( minper!= 1 or maxper != 99):
        raise SyntaxError('norm and minper and maxper cannot be both defined')

    use_percentile = False
    if vmin is None or vmax is None:
        use_percentile = True

    if isinstance(img, (list, np.ndarray, tuple)):
        fig, axs = plt.subplots(1, len(img), figsize=figsize, dpi=dpi)
        
        if len(img) == 1:
            axs = [axs]

        if isinstance(title, str):
            fig.suptitle(title)

        for i in range(len(img)):

            if mask_center_rad is not None:
                 mask = circular_aperture(img[i], mask_center_rad, inverted=True)
                 img[i] = img[i] * mask

            if isinstance(title, (list, np.ndarray, tuple)):
                axs[i].set_title(title[i])
            else:
                axs[i].set_title(f'image {i + 1}')

            if use_percentile:
                vmax = np.nanpercentile(img[i], maxper)
                vmin = np.nanpercentile(img[i], minper)

            if norm is None:
                plot = axs[i].imshow(img[i], vmax=vmax, vmin=vmin, origin='lower', cmap=cmap)

            else:
                plot = axs[i].imshow(img[i], norm=norm, origin='lower', cmap=cmap)

            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            plt.colorbar(plot, ax=axs[i], fraction=0.046, pad=0.04)

            if not ticks:
                axs[i].set_xticks([])
                axs[i].set_yticks([])

            if show_center:
                axs[i].plot([img[i].shape[1] / 2 - .5], [img[i].shape[0] / 2 - .5], marker='x', color='red')

            if pixel_scale is not None:
                plot_pixel_scale(axs, pixel_scale, orientation)

            if orientation is not None:
                if isinstance(orientation, float):
                    plot_orientation(axs[i], orientation)
                elif isinstance(orientation, list):
                    plot_orientation(axs[i], orientation[i])
                else:
                    raise ValueError('orientation can be either "float" or "list" of floats with'
                                     ' same dimensions als image list')

    return




if __name__ == '__main__':
    from astropy.io import fits
    from os.path import join, basename

    img = fits.getdata('/data/beegfs/astro-storage/groups/beuther/pasteimle/ISPY_reduced_cubes_new/master_dark/master_dark_2023-03_2.0.fits')
    print(img.shape)
    show_img(img)
    plt.savefig('/home/pasteimle/code/ispy/SIRP/test.png')