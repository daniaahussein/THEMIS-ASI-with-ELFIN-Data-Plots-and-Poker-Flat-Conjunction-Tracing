import cdflib
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time

class ThemisPlotDataset:

    def __init__(self, datasetFileCDF: str):
        self.rawFile = datasetFileCDF
        self.dataset = cdflib.CDF(datasetFileCDF)
        self.info = self.dataset.cdf_info()

    def __str__(self):
        ret = ""
        for var in self.info.zVariables:
            ret+= var+"\n"
        return ret

    def _format_label(self, var: str, filename=False):
        # configure y-label:
        count = 0
        ylab = ""
        for char in var:
            if char == '_':
                count += 1
            elif count == 2:
                ylab += char

        # configure date of dataset based on cdf file
        ct = 0
        date = ""
        instrument = ""
        for char in self.rawFile:
            if char == '_':
                ct += 1
            elif ct == 2:
                instrument += char
            elif ct == 3:
                date += char

        xlabel = "Time"
        ylabel = ylab.upper()
        title = f"THEMIS-{var[2:3].upper()} {ylab.upper()}\n{instrument.upper()} {date}"

        if filename:
            file_name = f"THEMIS-{var[2:3].upper()}_{ylabel}_{instrument.upper()}_{date}.png"
            return xlabel, ylabel, title, file_name

        return xlabel, ylabel, title
    def plot_dataset(self, var: str, startrec=0, endrec=150, plot_dir = "themis_plots"):
        x = self.dataset.varget(var, startrec=startrec, endrec=endrec)
        y = self.dataset.varget(f"{var[0:8]}time", startrec=startrec, endrec=endrec)

        plt.figure()
        plt.plot(x, y, color='b')

        xlabel, ylabel, title, file_name= self._format_label(var, filename=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)

        filename = f"{plot_dir}/{file_name}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    def display_plot(self, map_display: str, var: str, vidName):
        # map_display can be any ThemisMapDataset object
        # generated mp4 file like a movie or mosaic movie
        fig = plt.figure()
        vid = cv2.VideoCapture(map_display)
        frame_number = 0

        fps = 5
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = f"{os.getcwd()}/mosaic_plots/{vidName}.mp4"
        video = cv2.VideoWriter(video_name, fourcc, fps, (640, 480))

        while True:
            # Reading frame from video or webcam
            # and resizing to match when combining them
            success, img = vid.read()
            img = cv2.resize(img, (640, 480))

            frame_number += 1
            x = self.dataset.varget(var, startrec=0, endrec=frame_number)
            y = self.dataset.varget(f"{var[0:8]}time", startrec=0, endrec=frame_number)

            # Plotting data
            plt.plot(x, y)
            xlabel, ylabel, title = self._format_label(var)
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)

            fig.canvas.draw()
            # converting matplotlib figure to Opencv image
            plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                 sep='')
            plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

            # Combining Original Frame and Plot Image
            result_img = np.hstack([img, plot])
            video.write(result_img)

            # Displaying the Combined Image:
            cv2.imshow("Image", result_img)
            cv2.waitKey(1)

            if success == False:
                break
        cv2.destroyAllWindows()
        video.release()
