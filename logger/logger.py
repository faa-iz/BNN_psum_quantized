import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
import os
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

def log_params(model, logger, epoch, single_layer_logging=False, save_numpy=False, save_dir=''):
    # restore real weights before logging
    for p in list(model.parameters()):
        if hasattr(p,'org'):
            p.data.copy_(p.org)
    #debug_print = {}
    if (save_numpy):
        save_dir = os.path.join(save_dir, 'Weights')
        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
    for tag, value in model.named_parameters():
        if (save_numpy):
            t = os.path.join(save_dir, tag)
            if (not os.path.exists(t)):
                os.mkdir(t)
        tag = tag.replace('.', '/')
        val = value.data.cpu().numpy()
        #debug_print[tag] = val.shape

        if (single_layer_logging or 'weight' not in tag or 'scale' in tag):
            if (save_numpy):
                    np.save(os.path.join(t, '{}.npy'.format(epoch)), val)
            logger.histo_summary(tag, val, epoch, bins=val.shape[0])
        else:
            if (len(val.shape) == 4):
                for i in range(0, val.shape[0]):
                    if (save_numpy):
                        tt = os.path.join(t, str(i))
                        if (not os.path.exists(tt)):
                            os.mkdir(tt)
                        np.save(os.path.join(tt, '{}.npy'.format(epoch)), val[i])
                    logger.histo_summary(tag + str(i), val[i], epoch, bins=50)
            else:
                if (save_numpy):
                    import pdb; pdb.set_trace()
                    np.save(os.path.join(tt, '{}.npy'.format(epoch)), val)
                logger.histo_summary(tag, val, epoch, 50)

        if (value.grad is not None):
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
