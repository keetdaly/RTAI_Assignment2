import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

### Initializing the shared variables ###
Q1_res = defaultdict(int)
Q2_res = defaultdict(int)
Q3_res = defaultdict(int)

TOTAL_FRAMES = 1495

# Method to plot the bar graph for the F1 score and Throughput
def bar_plot(x_values, y_values, title, xlabel, ylabel):
    plt.bar(x_values, y_values, width = 0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title + ".png")
    plt.show()
    plt.close()

# Method to plot the line graph for each of the query
def plot(values, title, xlabel, ylabel):
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title + ".png")
    plt.show()
    plt.close()

# Method to calculating the F1 score for each of the query
def calculate_F1s():
    predicted = np.genfromtxt('predictions.csv', delimiter=',', skip_header=True)
    ground_truth = pd.read_excel('Ground Truth (Assignment 2).xlsx')
    ground_truth = ground_truth[1:]

    # Q1 data
    predicted_cars_frame = predicted[:, -1]
    gt_cars_frame = ground_truth.iloc[:, -1].to_numpy()

    # Q2 & Q3 data - Only considering rows (i.e frames) that have a car
    sedans_predict = np.squeeze(predicted[np.nonzero(gt_cars_frame), 1:6], axis=0)
    hatchback_predict = np.squeeze(predicted[np.nonzero(gt_cars_frame), 6:-1], axis=0)

    gt_sedans = np.squeeze(ground_truth.to_numpy().astype(float)[np.nonzero(gt_cars_frame), 1:6], axis=0)
    gt_hatchback = np.squeeze(ground_truth.to_numpy().astype(float)[np.nonzero(gt_cars_frame), 6:-1], axis=0)

    # Calculate F1 scores for Query 1
    for i in range(len(gt_cars_frame)-1):
        predicted = predicted_cars_frame[i]
        gt = gt_cars_frame[i]
        if predicted < gt:
            # Detected less cars than in frame
            # Difference = missed cars, FN up
            Q1_res["FN"] += gt - predicted
            # Predicted = correct cars in frame, TP up
            Q1_res["TP"] += predicted
        elif predicted > gt:
            # Detected more cars than in frame
            # Difference = extra cars, FP up
            Q1_res["FP"] += predicted - gt
            # GT = correct cars, TP up
            Q1_res["TP"] += gt
        elif predicted == gt and predicted != 0:
            # Car(s) correctly deteced
            Q1_res["TP"] += predicted
        else:
            # No car detected correctly
            Q1_res["TN"] += 1

    P1 = Q1_res["TP"]/(Q1_res["TP"] + Q1_res["FP"])
    R1 = Q1_res["TP"]/(Q1_res["TP"] + Q1_res["FN"])
    F11 = round((2 * P1 * R1)/(P1 + R1),2)

    # Calculate F1 scores for Query 2
    rows, cols = sedans_predict.shape

    for i in range(rows):
        p_sedans = sedans_predict[i]
        # Num predicted Sedans
        num_p_sedans = np.sum(p_sedans)
        p_hatchbacks = hatchback_predict[i]
        # Num predicted Hatchbacks
        num_p_hatchbacks = np.sum(p_hatchbacks)
        gt_sedan = gt_sedans[i]
        # Num GT Sedans
        num_gt_sedans = np.sum(gt_sedan)
        gt_hatchbacks = gt_hatchback[i]
        # Num GT Hatchbacks
        num_gt_hatchbacks = np.sum(gt_hatchbacks)

        if num_p_sedans == num_gt_sedans:
            if num_p_sedans == 0:
                Q2_res["TN"] += 1
            else:
                # TP goes up by num detected cars
                Q2_res["TP"] += num_p_sedans
        else:
            if num_p_sedans < num_gt_sedans:
                Q2_res["FN"] += num_gt_sedans - num_p_sedans
                Q2_res["TP"] += num_p_sedans
            else:
                Q2_res["FP"] += num_p_sedans - num_gt_sedans
                Q2_res["TP"] += num_gt_sedans

        if num_p_hatchbacks == num_gt_hatchbacks:
            if num_p_hatchbacks == 0:
                Q2_res["TN"] += 1
            else:
                Q2_res["TP"] += num_p_hatchbacks
        else:
            if num_p_hatchbacks < num_gt_hatchbacks:
                Q2_res["FN"] += num_gt_hatchbacks - num_p_hatchbacks
                Q2_res["TP"] += num_p_hatchbacks
            else:
                Q2_res["FP"] += num_p_hatchbacks - num_gt_hatchbacks
                Q2_res["TP"] += num_gt_hatchbacks


    P2 = Q2_res["TP"]/(Q2_res["TP"] + Q2_res["FP"])
    R2 = Q2_res["TP"]/(Q2_res["TP"] + Q2_res["FN"])
    F12 = round((2 * P2 * R2)/(P2 + R2),2)

    #  Calculate F1 scores for Query 3
    for i in range(rows):
        p_sedans = sedans_predict[i]
        p_hatchbacks = hatchback_predict[i]
        gt_sedan = gt_sedans[i]
        gt_hatchbacks = gt_hatchback[i]

        # Check Sedans
        for j in range(5):
            sedan_p = p_sedans[j]
            sedan_gt = gt_sedan[j]
            if sedan_p == sedan_gt:
                if sedan_p == 0:
                    Q3_res["TN"] += 1
                else:
                    Q3_res["TP"] += sedan_p
            else:
                if sedan_p < sedan_gt:
                    Q3_res["FN"] += sedan_gt - sedan_p
                    Q3_res["TP"] += sedan_p
                else:
                    Q3_res["FP"] += sedan_p - sedan_gt
                    Q3_res["TP"] += sedan_gt

        # Check hatchbacks
        for j in range(5):
            hatchback_p = p_hatchbacks[j]
            hatchback_gt = gt_hatchbacks[j]
            if hatchback_p == hatchback_gt:
                if hatchback_p == 0:
                    Q3_res["TN"] += 1
                else:
                    Q3_res["TP"] += hatchback_p
            else:
                if hatchback_p < hatchback_gt:
                    Q3_res["FN"] += hatchback_gt - hatchback_p
                    Q3_res["TP"] += hatchback_p
                else:
                    Q3_res["FP"] += hatchback_p - hatchback_gt
                    Q3_res["TP"] += hatchback_gt

    P3 = Q3_res["TP"]/(Q3_res["TP"] + Q3_res["FP"])
    R3 = Q3_res["TP"]/(Q3_res["TP"] + Q3_res["FN"])
    F13 = round((2 * P3 * R3)/(P3 + R3),2)

    return [F11, F12, F13]

def init_extraction_times():
    # Extraction times
    Q1_times = np.genfromtxt("Q1_extraction_times.csv", delimiter=",")
    E2_times = np.genfromtxt("Q2_extraction_times.csv", delimiter=",")

    # Query 2: sum of times of Q1 and E2 which is time to classify car type
    Q2_times = Q1_times + E2_times
    E3_times = np.genfromtxt("Q3_extraction_times.csv", delimiter=",")

    # Query 3: sum of times of Query 2 and E3 which is time to classify colour
    Q3_times = Q2_times + E3_times
    return (Q1_times, Q2_times, Q3_times)

# Method to calculate the throughput for each of the query
def calculate_throughputs(Q1_times, Q2_times, Q3_times):
    total_time_Q1 = np.sum(Q1_times) / 1000
    total_time_Q2 = np.sum(Q2_times) / 1000
    total_time_Q3 = np.sum(Q3_times) / 1000

    throughput_Q1 = round(TOTAL_FRAMES / total_time_Q1, 2)
    throughput_Q2 = round(TOTAL_FRAMES / total_time_Q2, 2)
    throughput_Q3 = round(TOTAL_FRAMES / total_time_Q3, 2)

    return [throughput_Q1, throughput_Q2, throughput_Q3]

if __name__ == "__main__":
    # Initializing extraction times for performance evaluation
    Q1_times, Q2_times, Q3_times = init_extraction_times()
    ETs = (np.sum(Q1_times)/1000, np.sum(Q2_times)/1000, np.sum(Q3_times)/1000)
    print("Query Extraction Times: ",ETs)

    # Computing the F1 score and throughput for all the queries
    queries = ["Query 1", "Query 2", "Query 3"]
    F1s = calculate_F1s()
    print("F1 scores: ",F1s)
    Ts = calculate_throughputs(Q1_times, Q2_times, Q3_times)
    print("Query Throughputs: ",Ts)

    # Plotting the F1 score and throughput in bar graph
    bar_plot(queries, F1s, "F1_scores", "Queries", "F1 Score")
    bar_plot(queries, ETs, "Extraction Times", "Queries", "Extraction Time")
    bar_plot(queries, Ts, "Throughputs", "Queries", "Throughput (frame/s)")

    # Plotting the line graph for the event extraction time for
    plot(Q1_times/1000, "Query1_times_all_frames", "Frame Num", "Time (s)")
    plot(Q2_times/1000, "Query2_times_all_frames", "Frame Num", "Time (s)")
    plot(Q3_times/1000, "Query3_times_all_frames", "Frame Num", "Time (s)")