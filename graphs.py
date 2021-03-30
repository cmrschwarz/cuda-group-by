#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import csv
import sys
import os

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
    "thread_per_group": "deepskyblue"
}
approach_markers = {
    "hashtable": "^",
    "hashtable_eager_out_idx": "+",
    "hashtable_lazy_out_idx": "*",
    "thread_per_group": "x"
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

def average_col(rows, col):
    class_cols = list(COLUMNS)
    class_cols.remove(col)
    output_rows = []
    for row_group in classify_mult(rows, class_cols).values():
        row = row_group[0]
        row[col] = col_average(row_group, col)
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


def throughput_over_group_size_barring_row_count(data):
    _, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("group count")
    ax.set_ylabel("throughput (GiB/s, 16 B per row)")
    rowcounts = sorted(classify(data, ROW_COUNT_COL).keys())
    rowcounts_str = ", ".join([str(rc) for rc in rowcounts])
    ax.set_title(f"Throughput over Group Count, best in class\nrowcounts: {rowcounts_str}")
    by_group_count = classify(data, GROUP_COUNT_COL)
    bar_width = 1.0 / (len(by_group_count) + 1)
    bar_gap = 0.07
    by_approaches = classify(data, APPROACH_COL)

    
    for ap_id, (ap, ap_rows) in enumerate(by_approaches.items()):
        prev_y_vals = None
        by_row_count = sorted(classify(ap_rows, ROW_COUNT_COL).items())
        for (rc, rc_rows) in by_row_count:
            by_group_count = classify(rc_rows, GROUP_COUNT_COL)
            # fixed approach, row count and group count
            # averaged iterations
            # --> best in class over grid dim, block dim and stream count  

            best_in_class = sort_by_col(
                highest_in_class(by_group_count, THROUGHPUT_COL).values(), 
                GROUP_COUNT_COL
            )
            y_vals = col_vals(best_in_class,THROUGHPUT_COL)
            y_bar_vals = list(y_vals)
            if prev_y_vals is not None:
                for i in range(0, len(prev_y_vals)):
                    y_bar_vals[i] -= prev_y_vals[i] * (1 + bar_gap)
            x_positions = [
                i + (ap_id - len(by_group_count) / 2. + 1) * bar_width 
                for i in range(0, len(y_vals))
            ]
            
            ax.bar(
                x_positions, y_bar_vals, bar_width, 
                label = f"{ap}" if prev_y_vals is None else None,
                bottom=[y * (1 + bar_gap) for y in prev_y_vals] if prev_y_vals is not None else 0,
                color=approach_colors[ap]
            )
            ax.set_xticks(range(0, len(x_positions)))
            ax.set_xticklabels(col_vals(best_in_class, GROUP_COUNT_COL))
            prev_y_vals = y_vals
    ax.legend()
    ax.set_yscale("log")
    plt.savefig(f"throughput_over_group_size_barring_row_count.png")


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
    data = average_col(data, ITERATION_COL)

    #generate graphs
    os.chdir(output_path)
    throughput_over_group_count(data)
    throughput_over_stream_count(data, 32)
    throughput_over_group_size_barring_row_count(data)

if __name__ == "__main__":
    main()
