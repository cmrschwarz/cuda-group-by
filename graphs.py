#!/usr/bin/env python3
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
import sys
import multiprocessing
import os
import time
import math
import numpy as np

#constants
DB_ROW_SIZE_BYTES = 16

#columns
APPROACH_COL = 0
GROUP_COUNT_COL = 1
ROW_COUNT_COL = 2
GRID_DIM_COL = 3
BLOCK_DIM_COL = 4
STREAM_COUNT_COL = 5
ITERATION_COL = 6
TIME_MS_COL = 7
THROUGHPUT_COL = 8
COLUMN_COUNT = 9
#just throughput for now
VIRTUAL_COLUMN_COUNT = 1

COLUMNS = list(range(0, COLUMN_COUNT))

#coloring options
legacy_approach_remap = {
    "thread_per_group": "block_cmp",
    "thread_per_group_hashmap_writeout": "block_cmp",
    "thread_per_group_naive_writeout": "block_cmp_old_naive_writeout",
    "threads_per_group": "warp_cmp",
    "hashtable_lazy_out_idx": "hashtable",
}
approach_colors = {
    "hashtable": "darkorange",
    "hashtable_eager_out_idx": "green",
    "warp_cmp": "gold",
    "block_cmp": "deepskyblue",
    "block_cmp_old": "navy",
    "block_cmp_old_naive_writeout": "darkred",
    "shared_mem_hashtable": "purple",
    "cub_radix_sort": "turquoise",
    "throughput_test": "lightgray",
    "per_thread_hashtable": "teal",
    "per_thread_hashtable_bank_optimized": "lime",
}
approach_markers = {
    "hashtable": "^",
    "hashtable_eager_out_idx": "^",
    "hashtable_lazy_out_idx": "+",
    "warp_cmp": "<",
    "block_cmp": "x",
    "block_cmp_old": "1",
    "block_cmp_old_naive_writeout": "*",
    "shared_mem_hashtable": ">",
    "cub_radix_sort": "o",
    "throughput_test": "+",
    "per_thread_hashtable": "2",
    "per_thread_hashtable_bank_optimized": "3",
}



#helper functions

def classify(data, classifying_col):
    classes = {}
    for row in data:
        c = row[classifying_col]
        if c not in classes:
            classes[c] = [row]
        else:
            classes[c].append(row)
    return classes

def classify_mult(data, classifying_cols):
    classes = {}
    for row in data:
        c = []
        for cc in classifying_cols:
            c.append(row[cc])
        c = tuple(c)
        if c not in classes:
            classes[c] = [row]
        else:
            classes[c].append(row)
    return classes


def highest_in_class(classes, maximize_col):
    highest = {}
    for k,v in classes.items():
        highest[k] = max(v, key=lambda r: r[maximize_col]) 
    return highest

def lowest_in_class(classes, minimize_col):
    lowest = {}
    for k,v in classes.items():
        lowest[k] = min(v, key=lambda r: r[minimize_col]) 
    return lowest

def col_vals_l(rows, col):
    return map(lambda r: r[col], rows)

def col_vals(rows, col):
    return list(col_vals_l(rows, col))

def unique_col_vals(rows, col):
    results = {}
    for r in rows:
        results[r[col]] = True
    return list(results.keys())

def col_average(rows, col):
    return sum(col_vals_l(rows, col)) / len(rows)

def average_columns(rows, cols):
    class_cols = list(COLUMNS)
    for c in cols:
        class_cols.remove(c)

    output_rows = []
    for row_group in classify_mult(rows, class_cols).values():
        row = row_group[0]
        for c in cols:
            row[c] = col_average(row_group, c)
        output_rows.append(row)
    return output_rows

def class_with_lowest_average(classes, avg_col):
    return min(
        classes.items(), 
        key=lambda kv: col_average(kv[1], avg_col)
    )[0]

def class_with_highest_average(classes, avg_col):
    return max(
        classes.items(), 
        key=lambda kv: col_average(kv[1], avg_col)
    )[0]

def highest_class_average(classes, avg_col):
    return max(
        [(col_average(rows, avg_col), cl) for (cl, rows) in classes.items()]
    )


def sort_by_col(rows, sort_col):
    return sorted(rows, key=lambda r: r[sort_col])

def max_col_val(rows, col):
    return max(rows, key=lambda r: r[col])[col]

def min_col_val(rows, col):
    return min(rows, key=lambda r: r[col])[col]


def filter_col_val(rows, col, val):
    return list(filter(lambda r: r[col] == val, rows))


# graph generators

def throughput_over_group_count(data, log=False):
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("group count")
    ax.set_ylabel("throughput (GiB/s, 16 B per row)")
    max_rowcount = max_col_val(data, ROW_COUNT_COL)
    ax.set_title(f"Throughput over Group Count (rowcount = {max_rowcount}, best in class)")

    rowcount_filtered = filter_col_val(data, ROW_COUNT_COL, max_rowcount)

    by_approaches = classify(rowcount_filtered, APPROACH_COL)

    for approach, rows in by_approaches.items():
        by_group_count = classify(rows, GROUP_COUNT_COL)
        group_counts = sorted(unique_col_vals(rows, GROUP_COUNT_COL))
        x = group_counts
        y = []
        for gc in group_counts:
            gc_rows = by_group_count[gc]
            classes = classify_mult(gc_rows, [ROW_COUNT_COL, GROUP_COUNT_COL, STREAM_COUNT_COL])
            avg, _ = highest_class_average(classes, THROUGHPUT_COL)
            y.append(avg)

        ax.plot(
            x,y,
            marker=approach_markers[approach],
            color=approach_colors[approach],
            markerfacecolor='none',
            label=f"{approach}", alpha=0.7)
    ax.set_xscale("log", basex=2)
    ax.set_xticks(unique_col_vals(data, GROUP_COUNT_COL))
    if log:
        ax.set_yscale("log", basey=2)
    else:
        ax.set_ylim(0)
    ax.legend()
    fig.savefig("throughput_over_group_count" + ("_log" if log else "") +".png")

def throughput_over_stream_count(data, group_count):
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("stream count")
    ax.set_ylabel("throughput (GiB/s, 16 B per row)")
    max_rowcount = max_col_val(data, ROW_COUNT_COL)
    ax.set_title(f"Throughput over Stream Count (row_count = {max_rowcount}, group_count={group_count}, best in class)")

    rowcount_filtered = filter_col_val(data, ROW_COUNT_COL, max_rowcount)
    groupcount_filtered = filter_col_val(rowcount_filtered, GROUP_COUNT_COL, group_count)

    by_approaches = classify(groupcount_filtered, APPROACH_COL)

    for approach, rows in by_approaches.items():
        lines = classify_mult(rows, [GRID_DIM_COL, BLOCK_DIM_COL])
        best_line_class = class_with_highest_average(lines, THROUGHPUT_COL)
        best_line = lines[best_line_class]
        best_line = sort_by_col(best_line, STREAM_COUNT_COL)
        x = col_vals(best_line, STREAM_COUNT_COL)
        y = col_vals(best_line, THROUGHPUT_COL)
        ax.plot(
            x,y,
            marker=approach_markers[approach],
            color=approach_colors[approach],
            markerfacecolor='none',
            label=f"{approach}, (gd,bd) = {best_line_class}")
    
    ax.set_xticks(unique_col_vals(data, STREAM_COUNT_COL))
    ax.set_ylim(0)
    ax.legend()
    fig.savefig(f"throughput_over_stream_count_gc{group_count}.png")


def grid_dim_block_dim_heatmap(data, approach, group_count=None, stream_count=None):
    timestamp("heatmap start")
    assert((group_count is None) != (stream_count is None)) 
    filtered = filter_col_val(data, APPROACH_COL, approach)

    if stream_count is not None:
        y_axis_col = GROUP_COUNT_COL
        filtered = filter_col_val(filtered, STREAM_COUNT_COL, stream_count)
    else:
        y_axis_col = STREAM_COUNT_COL
        filtered = filter_col_val(filtered, GROUP_COUNT_COL, group_count)

    rowcount_vals = sorted(dict.fromkeys(col_vals(filtered, ROW_COUNT_COL)))

    y_axis_vals = sorted(dict.fromkeys(col_vals(filtered, y_axis_col)))
    
    fig, axes = plt.subplots(
        nrows=len(rowcount_vals), ncols=len(y_axis_vals),
        dpi=200, 
        figsize=(
            (len(y_axis_vals) + 1) * len(dict.fromkeys(col_vals(filtered, GRID_DIM_COL))),
            len(rowcount_vals) * len(dict.fromkeys(col_vals(filtered, BLOCK_DIM_COL)))
        ),
    )
    for (row_id, rc) in enumerate(rowcount_vals):
        by_row_count = filter_col_val(filtered, ROW_COUNT_COL, rc)
        rc_min_val = min_col_val(by_row_count, THROUGHPUT_COL)
        rc_max_val = max_col_val(by_row_count, THROUGHPUT_COL)
        rc_val_range = rc_max_val - rc_min_val
        for (col_id, yv) in enumerate(y_axis_vals):
            ax = axes[row_id][col_id]
            by_y_val = filter_col_val(by_row_count, y_axis_col, yv)
            classified = classify_mult(by_y_val, [GRID_DIM_COL, BLOCK_DIM_COL])
            if len(classified) == 0: continue
            gd_vals, bd_vals = zip(*classified.keys())
            gd_vals = sorted(dict.fromkeys(gd_vals).keys())
            # reverse the y axis so the small values are at the bottom
            bd_vals = sorted(dict.fromkeys(bd_vals).keys(), reverse=True)
            gd_indices = {v:k for (k,v) in enumerate(gd_vals)}
            bd_indices = {v:k for (k,v) in enumerate(bd_vals)}

        
            ax.set_xlabel("grid dim")
            ax.set_ylabel("block dim")
            ax.set_title(
                f"rc = {rc}, " 
                + (f"sc = {yv}" if stream_count is None else f"gc = {yv}")
            )
        
            grid = [[0] * len(gd_vals) for i in range(len(bd_vals))]
            vals = []
            for (gd, bd), rows in classified.items():
                tp = col_average(rows, THROUGHPUT_COL)
                grid[bd_indices[bd]][gd_indices[gd]] = tp
                vals.append(tp)
            for x in range(len(gd_vals)):
                for y in range(len(bd_vals)):
                    val = grid[y][x]
                    if val < 1000: 
                        val_str = str(round(val, int(3-max(math.log10(val), 0))))
                    else:
                        val_str = str(round(val))

                    ax.text(
                        x, y, val_str, ha="center", va="center", 
                        color="black" if (val - rc_min_val) / rc_val_range < 0.8 else "white" 
                    )
            im = ax.imshow(np.array(grid), cmap='Reds', vmin=rc_min_val, vmax=rc_max_val)
            ax.set_xticks(range(len(gd_vals)))
            ax.set_yticks(range(len(bd_vals)))

            ax.set_xticklabels(gd_vals)
            ax.set_yticklabels(bd_vals)
            if col_id + 1 == len(y_axis_vals):
                cax = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.5)
                fig.colorbar(im, cax=cax)
    fig.tight_layout(h_pad=2)
    fig.savefig(
        f"{approach}_grid_block_heatmap_over_rowcount_and_" 
        + (f"group_count_sc_{stream_count}" if stream_count is not None else f"stream_count_gc{group_count}")
        + ".png",
        bbox_inches="tight")

def col_stddev_over_row_count(data, group_count, relative, minimize, col, col_str, col_unit=None):
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("row count")
    if relative:
        ax.set_ylabel(f"relative {col_str} standard deviation in percent")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        ax.set_ylabel(
            f"{col_str} standard deviation" 
            + (f" ({col_unit})" if col_unit is not None else "")
        )
    groupcount_filtered = filter_col_val(data, GROUP_COUNT_COL, group_count)

    ax.set_title(
        ("Relative " if relative else "") 
        + f"{col_str.capitalize()} Standard Deviation over Row Count" 
        + f" (group_count={group_count}, " 
        + ("lowest " if minimize else "highest")
        + f" {col_str} in class)"
    )

    by_approaches = classify(groupcount_filtered, APPROACH_COL)

    for approach, rows in by_approaches.items():
        classified = classify_mult(rows, [GRID_DIM_COL, BLOCK_DIM_COL, STREAM_COUNT_COL])
        if minimize:
            best_class = class_with_lowest_average(classified, col)
        else:
            best_class = class_with_highest_average(classified, col)

        by_row_count = classify(classified[best_class], ROW_COUNT_COL)
        row_counts = sorted(by_row_count.keys())
        stddevs = []
        for rc in row_counts:
            rc_rows = by_row_count[rc]
            avg = col_average(rc_rows, col)
            dev = 0
            for r in rc_rows:
                dev += (r[col] - avg)**2
            stddev = math.sqrt(dev / len(rc_rows))
            if relative:
                stddev = (stddev / avg) * 100
            stddevs.append(stddev)

        ax.plot(
            row_counts,stddevs,
            marker=approach_markers[approach],
            color=approach_colors[approach],
            markerfacecolor='none',
            label=f"{approach} (gd,bd,sc)={best_class}")
    
    ax.set_ylim(0)
    ax.set_xscale("log", basex=2)
    ax.set_xticks(unique_col_vals(data, ROW_COUNT_COL))
    ax.legend()
    fig.savefig(
        ("relative_" if relative else "") 
        + f"{col_str}_stddev_over_row_count_gc{group_count}.png"
    )


def runtime_over_group_size_barring_approaches_stacking_row_count(data):
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("group count")
    ax.set_ylabel("runtime (ms)")
    rowcounts = sorted(classify(data, ROW_COUNT_COL).keys())
    rowcounts_str = ", ".join([str(rc) for rc in rowcounts])
    ax.set_title(f"Runtime over Group Count, best in class\nrowcounts: {rowcounts_str}\n")
    by_approaches = classify(data, APPROACH_COL)
    approach_count = len(by_approaches)
    bar_width = 1.0 / (approach_count + 1)
    graph_max_value = max_col_val(data, TIME_MS_COL)
    graph_min_value = min_col_val(data, TIME_MS_COL)
    graph_height = math.log(graph_max_value, 10) - math.log(graph_min_value, 10)
    bar_gap = 0.005 * graph_height
    bar_count_per_group_count = {}
    bar_index_per_group_count = {}
    i = 0
    for gc, rows in sorted(classify(data, GROUP_COUNT_COL).items()):
        bar_count_per_group_count[gc] = len(classify(rows, APPROACH_COL))
        bar_index_per_group_count[gc] = i
        i += 1
    
    for ap_id, (ap, ap_rows) in enumerate(by_approaches.items()):
        prev_y_vals = []
        by_row_count = sorted(classify(ap_rows, ROW_COUNT_COL).items())
        for (_, rc_rows) in by_row_count:
            by_group_count = classify(rc_rows, GROUP_COUNT_COL)
            # fixed approach, row count and group count
            # averaged iterations
            # --> best in class over grid dim, block dim and stream count  

            best_in_class = sort_by_col(
                lowest_in_class(by_group_count, TIME_MS_COL).values(), 
                GROUP_COUNT_COL
            )
            group_counts = col_vals(best_in_class, GROUP_COUNT_COL)
            y_vals = col_vals(best_in_class,TIME_MS_COL)
            y_bar_vals = list(y_vals)
            x_positions = [
                bar_index_per_group_count[gc] + (ap_id - bar_count_per_group_count[gc] / 2. + 0.5) * bar_width 
                for gc in group_counts
            ]
            bottoms = 0
            if prev_y_vals != []:
                bottoms = [y * (10 ** bar_gap) for y in prev_y_vals] 
                delc = 0
                for i in range(0, len(prev_y_vals)):
                    diff = math.log(y_vals[i], 10) - (math.log(prev_y_vals[i], 10) + bar_gap)
                    if  diff < bar_gap * 0.5:
                        del x_positions[i-delc]
                        y_vals[i] = prev_y_vals[i]
                        del y_bar_vals[i - delc]
                        del bottoms[i - delc]
                        delc+=1
                    else:
                        y_bar_vals[i - delc] -= prev_y_vals[i] * (10 ** bar_gap)
            if x_positions != []:
                col = approach_colors[ap]
                ax.bar(
                    x_positions, y_bar_vals, bar_width, 
                    label = f"{ap}" if prev_y_vals == [] else None,
                    bottom=bottoms,
                    color=col
                )
            prev_y_vals = y_vals
    ax.set_xticks(range(0, len(bar_count_per_group_count)))
    ax.set_xticklabels(sorted(bar_index_per_group_count.keys()))
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(f"throughput_over_group_size_barring_approaches_stacking_row_count.png")

def throughput_over_group_size_barring_row_count_stacking_approaches(data, logscale):
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("group count")
    ax.set_ylabel("throughput (GiB/s, 16 B per row)")
    #rowcounts = sorted(classify(data, ROW_COUNT_COL).keys())
    #rowcounts_str = ", ".join([str(rc) for rc in rowcounts])
    log_base = 10
    if logscale:
        transform = lambda x: math.log(x, log_base) 
    else:
        transform = lambda x: x

    graph_max_value = max_col_val(data, THROUGHPUT_COL)
    graph_min_value = min_col_val(data, THROUGHPUT_COL)
    # increase space at the top so the row count labels don't overflow 
    if logscale:
        graph_max_value *= 10**(0.5 * math.log(graph_max_value, log_base))
    else:
        graph_max_value *= 1.2  


 
    graph_height = (transform(graph_max_value) - transform(graph_min_value))
    text_offset = 0.01
    split_diff = 0.01 * graph_height
    ax.set_title(
        "Throughput over Group Count, Bars per Row Count, best in class\n" 
        + f"merge criterium: difference" 
        + (" of logs" if logscale else "") 
        + f" < {split_diff:.5f}\n"
    )
    by_row_count = classify(data, ROW_COUNT_COL)
    n_row_counts = len(by_row_count)
    bar_width = 1.0 / (n_row_counts + 1)
    bar_gap=bar_width * 0.1
    bar_width -= bar_gap
   
    bar_count_per_group_count = {}
    bar_index_per_group_count = {}
    by_group_count = classify(data, GROUP_COUNT_COL)
    i = 0
    for gc, rows in sorted(by_group_count.items()):
        bar_count_per_group_count[gc] = len(classify(rows, ROW_COUNT_COL))
        bar_index_per_group_count[gc] = i
        i += 1

    chart_data_per_group_count = {}
    for gc, gc_rows in by_group_count.items():
        approach_vals_by_row_count = {}
        for rc in by_row_count.keys():
            approach_vals_by_row_count[rc] = []

        by_approaches = sorted(classify(gc_rows, APPROACH_COL).items())
        for (ap, ap_rows) in by_approaches:
            ap_by_row_count = classify(ap_rows, ROW_COUNT_COL)
            # fixed approach, row count and group count
            # averaged iterations
            # --> best in class over grid dim, block dim and stream count  
            for rc, row in highest_in_class(ap_by_row_count, THROUGHPUT_COL).items():
                approach_vals_by_row_count[rc].append((ap, row[THROUGHPUT_COL]))
        # sort approaches for each row count -> bar
        for rc, ap_vals_of_rc in approach_vals_by_row_count.items():
            ap_vals_of_rc.sort(key=lambda ap_tp_tup: ap_tp_tup[1])
            # group approaches differing very little so we can split the bar
            last_base_val_log = transform(ap_vals_of_rc[0][1])
            last_base_idx = 0
            ap_groups_of_rc = []
            for i in range(0, len(ap_vals_of_rc)):
                next_in_bounds = (i+1 < len(ap_vals_of_rc))
                val_log = 0 if not next_in_bounds else transform(ap_vals_of_rc[i+1][1])
                if (
                    not next_in_bounds
                    or (val_log - last_base_val_log) >= split_diff
                ):
                    group = ap_vals_of_rc[last_base_idx:i+1]
                    (aps, tps) = zip(*group)
                    ap_groups_of_rc.append((
                        {ap: idx for idx, ap in enumerate(sorted(aps))},
                        sum(tps) / len(tps)
                    ))
                    if next_in_bounds:
                        last_base_idx = i+1
                        last_base_val_log = val_log
            approach_vals_by_row_count[rc] = ap_groups_of_rc

        chart_data_per_group_count[gc] = approach_vals_by_row_count


    
    # sort bar groups by group count
    chart_data_per_group_count = sorted(chart_data_per_group_count.items())

    for ap in classify(data, APPROACH_COL).keys():
        x_vals = []
        y_vals = []
        widths = []
        bottoms = []
        for gc, approach_vals_by_row_count in chart_data_per_group_count:
            for rc_index, (rc, ap_vals_of_rc) in enumerate(sorted(approach_vals_by_row_count.items())):
                share_count = 1
                shared_idx = 0
                for i, (aps, _) in enumerate(ap_vals_of_rc):
                    if ap in aps:
                        share_count = len(aps) 
                        shared_idx = aps[ap]
                        idx = i
                        break
                else:
                    continue
                x_vals.append( 
                    bar_gap * 0.5 +
                    bar_index_per_group_count[gc] 
                    + (rc_index - bar_count_per_group_count[gc] / 2. + 0.5) * (bar_width + bar_gap)  
                    + (shared_idx - share_count / 2 + 0.5) * (bar_width / share_count)
                )
                bottom = ap_vals_of_rc[idx-1][1] if idx > 0 else 0
                y_vals.append(ap_vals_of_rc[idx][1] - bottom)
                widths.append(bar_width / share_count)
                bottoms.append(bottom)
        ax.bar(
            x_vals, y_vals, widths, 
            label = f"{ap}",
            bottom=bottoms,
            color=approach_colors[ap]
        )
    for gc, approach_vals_by_row_count in chart_data_per_group_count:
        for rc_index, (rc, ap_vals_of_rc) in enumerate(sorted(approach_vals_by_row_count.items())):
            xpos = (
                0.5 * bar_gap + bar_index_per_group_count[gc] 
                + (rc_index - bar_count_per_group_count[gc] / 2. + 0.5) * (bar_width + bar_gap)
            )
            ypos = ap_vals_of_rc[-1][1] 
            if logscale:
                ypos *= 10**(text_offset * math.log(graph_max_value, log_base))
            else:
                ypos += text_offset * graph_max_value

            plt.annotate(str(rc), ha='center', va='bottom', rotation=90, xy=(xpos, ypos))
    ax.set_xticks(range(0, len(bar_count_per_group_count)))
    ax.set_xticklabels(sorted(bar_index_per_group_count.keys()))
    if logscale:
        ax.set_yscale("log", basey=log_base)
    ax.set_ylim(top=graph_max_value)
    ax.legend()
    fig.savefig(
        f"throughput_over_group_size_barring_row_count_stacking_approaches" 
        + ("_log" if logscale else "") 
        + ".png"
    )


def read_csv(path):
    data=[]
    with open(path) as file:
        reader = csv.reader(file, delimiter=';')
        header = next(reader) # skip header
        if len(header) == COLUMN_COUNT - VIRTUAL_COLUMN_COUNT:
            iters_fix = 0
        elif len(header) == COLUMN_COUNT - VIRTUAL_COLUMN_COUNT - 1:
            # legacy support for old benchmarks without iterations 
            iters_fix = 1
            # these old benchmarks only had the eager version and called
            # it 'hashtable'
            legacy_approach_remap["hashtable"] = "hashtable_eager_out_idx"
        else:
            raise ValueError("unexpected column count in " + path)

        for ap in approach_colors:
            if ap not in legacy_approach_remap:
                legacy_approach_remap[ap] = ap
        
        for csv_row in reader:
            data_row = [None] * COLUMN_COUNT
            data_row[APPROACH_COL] = legacy_approach_remap[csv_row[APPROACH_COL]]
            data_row[GROUP_COUNT_COL] = (int(csv_row[GROUP_COUNT_COL]))
            data_row[ROW_COUNT_COL] = (int(csv_row[ROW_COUNT_COL]))
            data_row[GRID_DIM_COL] = (int(csv_row[GRID_DIM_COL]))
            data_row[BLOCK_DIM_COL] = (int(csv_row[BLOCK_DIM_COL]))
            data_row[STREAM_COUNT_COL] = (int(csv_row[STREAM_COUNT_COL]))
            data_row[ITERATION_COL] = 0 if iters_fix != 0 else (int(csv_row[ITERATION_COL]))
            data_row[TIME_MS_COL] = (float(csv_row[TIME_MS_COL - iters_fix]))
            data_row[THROUGHPUT_COL] = (1000. * DB_ROW_SIZE_BYTES * data_row[ROW_COUNT_COL]) / data_row[TIME_MS_COL] / 2**30
            data.append(data_row)
    return data

def parallel(fns):
    processes=[]
    for fn in fns:
        p = multiprocessing.Process(target=fn)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def sequential(fns):
    for fn in fns:
        fn()


def timestamp(msg):
    global process_start_time
    print(
        "[" 
        + str(round(10**-6 * (time.time_ns() - process_start_time), 3)) 
        + " ms]: " 
        + msg
    ) 


def main():
    # initialization
    global process_start_time
    process_start_time = time.time_ns()

    #cli parsing
    args = sys.argv

    gen_all = False  
    if len(args) > 1 and args[1] == "-a":
        gen_all = True
        del args[1]

    if len(args) > 1:
        input_path = args[1]
    else:
        input_path="bench.csv"

    if len(args) > 2:
        output_path = args[2]
    else:
        output_path="./graphs"
        os.makedirs(output_path, exist_ok=True)
    
    #read in data
    data = read_csv(input_path)

    # average runs since we basically always need this
    data_avg = average_columns(
        data, 
        [ITERATION_COL, THROUGHPUT_COL, TIME_MS_COL]
    )

    #change to output path dir so the images are generated in the right folder
    os.chdir(output_path)

    #generate graphs (in parallel)
    jobs = [
        lambda: throughput_over_group_count(data_avg),
        lambda: throughput_over_group_count(data_avg, True),
        lambda: throughput_over_stream_count(data_avg, 32),
        lambda: col_stddev_over_row_count(data, 32, False, False, THROUGHPUT_COL, "throughput", "GiB/s, 16 B per row"),
        lambda: col_stddev_over_row_count(data, 32, True, False, THROUGHPUT_COL, "throughput"),
        lambda: col_stddev_over_row_count(data, 32, False, True, TIME_MS_COL, "runtime",  "time in ms"),
        lambda: col_stddev_over_row_count(data, 32, True, True, TIME_MS_COL, "runtime"),
        lambda: runtime_over_group_size_barring_approaches_stacking_row_count(data_avg),
        lambda: throughput_over_group_size_barring_row_count_stacking_approaches(data_avg, True),
        lambda: throughput_over_group_size_barring_row_count_stacking_approaches(data_avg, False),
    ]
    slow_jobs = [
        lambda: grid_dim_block_dim_heatmap(data_avg, "shared_mem_hashtable", stream_count=0),
    ]
    if(gen_all):
        jobs = jobs + slow_jobs
    
    parallel(jobs)

if __name__ == "__main__":
    main()
