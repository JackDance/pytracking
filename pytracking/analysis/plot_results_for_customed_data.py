import torch
import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt
from pytracking.evaluation import Tracker, get_dataset, trackerlist



def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)

def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect

def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect

def calc_err_center(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    return err_center


def calc_iou_overlap(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union

def check_eval_data_is_valid(eval_data, trackers, dataset):
    """ Checks if the pre-computed results are valid"""
    seq_names = [s.name for s in dataset]
    seq_names_saved = eval_data['sequences']

    tracker_names_f = [(t.name, t.parameter_name, t.run_id) for t in trackers]
    tracker_names_f_saved = [(t['name'], t['param'], t['run_id']) for t in eval_data['trackers']]

    return seq_names == seq_names_saved and tracker_names_f == tracker_names_f_saved

def get_plot_draw_styles():
    plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    return plot_draw_style

def merge_multiple_runs(eval_data):
    new_tracker_names = []
    ave_success_rate_plot_overlap_merged = []
    ave_success_rate_plot_center_merged = []
    ave_success_rate_plot_center_norm_merged = []
    avg_overlap_all_merged = []

    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all'])

    trackers = eval_data['trackers']
    merged = torch.zeros(len(trackers), dtype=torch.uint8)
    for i in range(len(trackers)):
        if merged[i]:
            continue
        base_tracker = trackers[i]
        new_tracker_names.append(base_tracker)

        match = [t['name'] == base_tracker['name'] and t['param'] == base_tracker['param'] for t in trackers]
        match = torch.tensor(match)

        ave_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].mean(1))
        ave_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].mean(1))
        ave_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].mean(1))
        avg_overlap_all_merged.append(avg_overlap_all[:, match].mean(1))

        merged[match] = 1

    ave_success_rate_plot_overlap_merged = torch.stack(ave_success_rate_plot_overlap_merged, dim=1)
    ave_success_rate_plot_center_merged = torch.stack(ave_success_rate_plot_center_merged, dim=1)
    ave_success_rate_plot_center_norm_merged = torch.stack(ave_success_rate_plot_center_norm_merged, dim=1)
    avg_overlap_all_merged = torch.stack(avg_overlap_all_merged, dim=1)

    eval_data['trackers'] = new_tracker_names
    eval_data['ave_success_rate_plot_overlap'] = ave_success_rate_plot_overlap_merged.tolist()
    eval_data['ave_success_rate_plot_center'] = ave_success_rate_plot_center_merged.tolist()
    eval_data['ave_success_rate_plot_center_norm'] = ave_success_rate_plot_center_norm_merged.tolist()
    eval_data['avg_overlap_all'] = avg_overlap_all_merged.tolist()

    return eval_data

def get_auc_curve(ave_success_rate_plot_overlap):
    # ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)

    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center):
    # ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]

    return prec_curve, prec_score

def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts):
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 13)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
              fontsize=font_size_legend, framealpha=1.0)

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()

def plot_draw_save_simple(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts):
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 13)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])

        plotted_lines.append(line[0])

        # tracker = trackers[id_sort]
        # disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(trackers, scores[id_sort]))

    ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
              fontsize=font_size_legend, framealpha=1.0)

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()

def get_tracker_display_name(tracker):
    if tracker['disp_name'] is None:
        if tracker['run_id'] is None:
            disp_name = '{}_{}'.format(tracker['name'], tracker['param'])
        else:
            disp_name = '{}_{}_{:03d}'.format(tracker['name'], tracker['param'],
                                              tracker['run_id'])
    else:
        disp_name = tracker['disp_name']

    return  disp_name


# 1. calculate pred_bb and anno_bb error
def calc_seq_err_robust_simple(pred_bb, anno_bb):
    pred_bb = pred_bb.clone()

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    return err_overlap, err_center, err_center_normalized

# 2. extract results
def extract_results_simple(result_plot_path, pred_txt_path, gt_txt_path, plot_bin_gap=0.05):
    """

    """
    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    avg_overlap_all = torch.zeros((1, 1), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((1, 1, threshold_set_overlap.numel()),
                                                dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((1, 1, threshold_set_center.numel()),
                                               dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((1, 1, threshold_set_center.numel()),
                                                    dtype=torch.float32)

    valid_sequence = torch.ones(1, dtype=torch.uint8)

    if os.path.isfile(pred_txt_path):
        pred_bb = torch.tensor(load_text(str(pred_txt_path), delimiter=('\t', ','), dtype=np.float64))
        anno_bb = torch.tensor(load_text(str(gt_txt_path), delimiter=('\t', ','), dtype=np.float64))
    else:
        raise Exception('Result not found. {}'.format(pred_txt_path))

    err_overlap, err_center, err_center_normalized = calc_seq_err_robust_simple(pred_bb, anno_bb)

    avg_overlap_all = err_overlap.mean()

    seq_length = anno_bb.shape[0]

    if seq_length <= 0:
        raise Exception('Seq length zero')

    ave_success_rate_plot_overlap[0, 0, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
    ave_success_rate_plot_center[0, 0, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
    ave_success_rate_plot_center_norm[0, 0, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    eval_data = {    'valid_sequence': valid_sequence.tolist(),
                     'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                     'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                     'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                     'avg_overlap_all': avg_overlap_all.tolist(),
                     'threshold_set_overlap': threshold_set_overlap.tolist(),
                     'threshold_set_center': threshold_set_center.tolist(),
                     'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data

# 3. plot results
def plot_results_simple(result_plot_path, tracker_names, plot_types=('success'),  **kwargs):
    plot_draw_styles = get_plot_draw_styles()

    print('\nPlotting results over sequence')
    print('\nGenerating plots for: {}'.format(result_plot_path))

    eval_data = extract_results_simple(result_plot_path, pred_txt_path, gt_txt_path)
    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap)
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100),
                             'title': 'Success plot'}
        plot_draw_save_simple(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path,
                       success_plot_opts)
    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center)
        threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}
        plot_draw_save_simple(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles,
                       result_plot_path,
                       precision_plot_opts)
        # ********************************  Norm Precision Plot **************************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center_norm)
        threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': (0, 100), 'title': 'Normalized Precision plot'}
        plot_draw_save_simple(prec_curve, threshold_set_center_norm, prec_score, tracker_names, plot_draw_styles,
                       result_plot_path,
                       norm_precision_plot_opts)
    # else:
    #     raise Exception('ERROR: plot_types should be success, prec or norm_prec')
    plt.show()


if __name__ == '__main__':

    # plot results
    pred_txt_path = r'/Users/jack/Downloads/basketball_results1/txt_results/dimp18.txt'
    gt_txt_path = r'/Users/jack/Downloads/basketball_results1/txt_results/groundtruth.txt'

    result_plot_path = '/Users/jack/Downloads/result_plot_path'
    tracker_names = 'dimp18'

    plot_results_simple(result_plot_path, tracker_names, plot_types=('norm_prec'))
