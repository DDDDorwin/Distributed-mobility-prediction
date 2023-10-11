from matplotlib import pyplot as plt
from preprocessing import Keys


def plot_loss(loss: list):
    """
    plot the loss of training
    :param loss list of loss in each epoch
    """

    plt.figure()

    plt.plot(loss)

    plt.savefig("loss.jpg")


def plot_test_graph(dataset, pred):
    """
    plot the graphs of comparison between the predictions and the test set
    generates 3 graphs : whole test set, subset of first 10 data points, subset of first 100 data points

    :param dataset: the original data frame
    :param pred: the predictions
    """

    # plot the data
    plt.grid(True)
    plt.plot(dataset.index[len(dataset) - len(pred):],
             dataset[Keys.INTERNET][len(dataset) - len(pred):])
    plt.plot(dataset.index[len(dataset) - len(pred):], pred)

    plt.savefig("prediction_with_test_set.jpg")

    plt.clf()

    test_start_index = len(dataset) - len(pred)
    plt.plot(dataset.index[test_start_index: test_start_index + 10],
             dataset[Keys.INTERNET][test_start_index: test_start_index + 10])
    plt.plot(dataset.index[test_start_index: test_start_index + 10], pred[:10])

    plt.savefig("test_subset_10_points.jpg")

    plt.clf()

    plt.plot(dataset.index[test_start_index: test_start_index + 100],
             dataset[Keys.INTERNET][test_start_index: test_start_index + 100])
    plt.plot(dataset.index[test_start_index: test_start_index + 100], pred[:100])

    plt.savefig("test_subset_100_points.jpg")
