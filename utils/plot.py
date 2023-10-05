from matplotlib import pyplot as plt
from preprocessing import Keys


def plot_test_graph(dataset, pred):
    """
    plot the graphs of comparison between the predictions and the test set
    generates 3 graphs : whole test set, subset of first 10 data points, subset of first 100 data points

    :param dataset: the original data frame
    :param pred: the predictions
    """

    # plot the data
    f, ax = plt.subplots()
    f.set_size_inches(20, 10)
    f.set_dpi(500)
    ax.grid(True)
    ax.plot(dataset.index[len(dataset) - len(pred):],
             dataset[Keys.INTERNET][len(dataset) - len(pred):], label='real_data')
    ax.plot(dataset.index[len(dataset) - len(pred):], pred, label='pred_data')
    ax.legend(loc='best')
    ax.set_title("comparison in test set")
    ax.xaxis_date()
    plt.xlabel("time")
    plt.ylabel("internet")
    plt.xticks(rotation=90)

    plt.savefig("prediction_with_test_set.jpg")

    plt.clf()

    f, ax = plt.subplots()
    ax.grid(True)
    test_start_index = len(dataset) - len(pred)
    ax.plot(dataset.index[test_start_index: test_start_index + 10],
             dataset[Keys.INTERNET][test_start_index: test_start_index + 10], label='real_data')
    ax.plot(dataset.index[test_start_index: test_start_index + 10], pred[:10], label='pred_data')
    ax.legend(loc='best')
    ax.set_title("comparison of first 10 points")
    ax.xaxis_date()
    plt.xlabel("time")
    plt.ylabel("internet")
    plt.xticks(rotation=90)

    plt.savefig("test_subset_10_points.jpg")

    plt.clf()

    f, ax = plt.subplots()
    f.set_size_inches(20, 10)
    f.set_dpi(500)
    ax.grid(True)
    ax.plot(dataset.index[test_start_index: test_start_index + 100],
             dataset[Keys.INTERNET][test_start_index: test_start_index + 100], label='real_data')
    ax.plot(dataset.index[test_start_index: test_start_index + 100], pred[:100], label='pred_data')
    ax.legend(loc='best')
    ax.set_title("comparison of first 100 points")
    ax.xaxis_date()
    plt.xlabel("time")
    plt.ylabel("internet")
    plt.xticks(rotation=90)

    plt.savefig("test_subset_100_points.jpg")

    def plot_whole_data(data, pred):
        # plt.plot([dataset.index[len(dataset)-len(pred)], dataset.index[len(dataset)-len(pred)]],
        #          [min(pred), max(pred)], 'k', label='real | pred')

        return None