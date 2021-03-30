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
TIME_MS_COL = 6
THROUGHPUT_COL = 7
COLUMN_COUNT = 8

#coloring options
approach_colors = {
    "hashtable": "darkorange",
    "thread_per_group": "deepskyblue"
}
approach_markers = {
    "hashtable": "^",
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

def col_vals(rows, col):
    return list(map(lambda r: r[col], rows))

def class_with_highest_average(classes, avg_col):
    return max(
        classes.items(), 
        key=lambda kv: sum(map(lambda r: r[avg_col], kv[1]))
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

def read_csv(path):
    data=[]
    with open(path) as file:
        reader = csv.reader(file, delimiter=';')
        next(reader) # skip header
        for csv_row in reader:
            data_row = [None] * COLUMN_COUNT
            data_row[APPROACH_COL] = (csv_row[APPROACH_COL])
            data_row[GROUP_COUNT_COL] = (int(csv_row[GROUP_COUNT_COL]))
            data_row[ROW_COUNT_COL] = (int(csv_row[ROW_COUNT_COL]))
            data_row[GRID_DIM_COL] = (int(csv_row[GRID_DIM_COL]))
            data_row[BLOCK_DIM_COL] = (int(csv_row[BLOCK_DIM_COL]))
            data_row[STREAM_COUNT_COL] = (int(csv_row[STREAM_COUNT_COL]))
            data_row[TIME_MS_COL] = (float(csv_row[TIME_MS_COL]))
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

    #generate graphs
    os.chdir(output_path)
    throughput_over_group_count(data)
    throughput_over_stream_count(data, 32)

if __name__ == "__main__":
    main()
