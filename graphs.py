#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import csv
import sys
import os
import math

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

COLUMNS = list(range(0, COLUMN_COUNT))

#coloring options
approach_colors = {
    "hashtable": "darkorange",
    "hashtable_eager_out_idx": "lightblue",
    "hashtable_lazy_out_idx": "green",
    "thread_per_group": "deepskyblue",
    "shared_mem_hashtable": "purple",
    "cub_radix_sort": "turquoise"
}
approach_markers = {
    "hashtable": "^",
    "hashtable_eager_out_idx": "+",
    "hashtable_lazy_out_idx": "*",
    "thread_per_group": "x",
    "shared_mem_hashtable": ">",
    "cub_radix_sort": "o"
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



def class_with_highest_average(classes, avg_col):
    return max(
        classes.items(), 
        key=lambda kv: col_average(kv[1], avg_col)
    )[0]

def sort_by_col(rows, sort_col):
    return sorted(rows, key=lambda r: r[sort_col])

def max_col_val(rows, col):
    return max(rows, key=lambda r: r[col])[col]

def min_col_val(rows, col):
    return min(rows, key=lambda r: r[col])[col]

def filter_col_val(rows, col, val):
    return list(filter(lambda r: r[col] == val, rows))


# graph generators

def throughput_over_group_count(data):
    _, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("group count")
    ax.set_ylabel("throughput (GiB/s, 16 B per row)")
    max_rowcount = max_col_val(data, ROW_COUNT_COL)
    ax.set_title(f"Throughput over Group Count (rowcount = {max_rowcount}, best in class)")

    rowcount_filtered = filter_col_val(data, ROW_COUNT_COL, max_rowcount)

    by_approaches = classify(rowcount_filtered, APPROACH_COL)

    for approach, rows in by_approaches.items():
        lines = classify_mult(rows, [GRID_DIM_COL, BLOCK_DIM_COL, STREAM_COUNT_COL])
        best_line_class = class_with_highest_average(lines, THROUGHPUT_COL)
        best_line = lines[best_line_class]
        best_line = sort_by_col(best_line, GROUP_COUNT_COL)
        x = col_vals(best_line, GROUP_COUNT_COL)
        y = col_vals(best_line, THROUGHPUT_COL)
        ax.plot(
            x,y,
            marker=approach_markers[approach],
            color=approach_colors[approach],
            markerfacecolor='none',
            label=f"{approach}, (gd,bd,sc) = {best_line_class}")
    ax.set_xscale("log", basex=2)
    #ax.set_yscale("log", basey=2)
    ax.set_ylim(0)
    ax.legend()
    plt.savefig("throughput_over_group_count.png")

def throughput_over_stream_count(data, group_count):
    _, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
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

    ax.set_xscale("log", basex=2)
    ax.set_ylim(0)
    ax.legend()
    plt.savefig(f"throughput_over_stream_count_gc{group_count}.png")


def runtime_over_group_size_barring_approaches_stacking_row_count(data):
    _, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
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
    bar_gap = 0.01 * graph_height
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
    plt.savefig(f"throughput_over_group_size_barring_approaches_stacking_row_count.png")

def throughput_over_group_size_barring_row_count_stacking_approaches(data, logscale):
    _, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("group count")
    ax.set_ylabel("throughput (GiB/s, 16 B per row)")
    #rowcounts = sorted(classify(data, ROW_COUNT_COL).keys())
    #rowcounts_str = ", ".join([str(rc) for rc in rowcounts])
    log_base = 10
    if logscale:
        transform = lambda x: math.log(x, log_base) 
    else:
        transform = lambda x: x

    graph_height = (
        transform(max_col_val(data, THROUGHPUT_COL))
        - transform(min_col_val(data, THROUGHPUT_COL))
    )
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
            by_row_count = classify(ap_rows, ROW_COUNT_COL)
            # fixed approach, row count and group count
            # averaged iterations
            # --> best in class over grid dim, block dim and stream count  
            for rc, row in highest_in_class(by_row_count, THROUGHPUT_COL).items():
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
            plt.annotate(
                str(rc), ha='center', va='bottom',
                xy=(
                    0.5 * bar_gap +
                    bar_index_per_group_count[gc] + (rc_index - bar_count_per_group_count[gc] / 2. + 0.5) * (bar_width + bar_gap),
                    ap_vals_of_rc[-1][1]
                )
            )
    ax.set_xticks(range(0, len(bar_count_per_group_count)))
    ax.set_xticklabels(sorted(bar_index_per_group_count.keys()))
    ax.legend()
    if logscale:
        ax.set_yscale("log", basey=log_base)
    plt.savefig(
        f"throughput_over_group_size_barring_row_count_stacking_approaches" 
        + ("_log" if logscale else "") 
        + ".png"
    )


def read_csv(path):
    data=[]
    with open(path) as file:
        reader = csv.reader(file, delimiter=';')
        header = next(reader) # skip header
        # legacy support for benchmarks without iterations 
        # we need to subtract 2 because of the virtual THROUGHPUT column
        iters_fix = 0 if len(header) != COLUMN_COUNT - 2 else 1
        
        for csv_row in reader:
            data_row = [None] * COLUMN_COUNT
            data_row[APPROACH_COL] = (csv_row[APPROACH_COL])
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


def main():
    #cli parsing
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path="bench.csv"
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path="./graphs"
        os.makedirs(output_path, exist_ok=True)
    
    #read in data
    data = read_csv(input_path)

    # average runs since we basically always need this
    data = average_columns(data, [ITERATION_COL, THROUGHPUT_COL, TIME_MS_COL])

    #generate graphs
    os.chdir(output_path)
    throughput_over_group_count(data)
    throughput_over_stream_count(data, 32)
    runtime_over_group_size_barring_approaches_stacking_row_count(data)
    throughput_over_group_size_barring_row_count_stacking_approaches(data, True)
    throughput_over_group_size_barring_row_count_stacking_approaches(data, False)

if __name__ == "__main__":
    main()
